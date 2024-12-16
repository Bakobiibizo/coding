import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import httpx
from bs4 import BeautifulSoup

class DirectoryMapper:
    def __init__(
        self,
        ignore_patterns: Optional[List[str]]=None,
        max_depth: int = 10,
        include_content: bool = False,
        content_extensions: Optional[List[str]]=None,
        ignore_hidden: bool = True
    ):
        self.ignore_patterns = ignore_patterns or ['node_modules', '.git', 'dist', 'build', '__pycache__']
        self.max_depth = max_depth
        self.include_content = include_content
        self.content_extensions = content_extensions  # Allow None to include all extensions
        self.ignore_hidden = ignore_hidden

    def should_ignore(self, name: str) -> bool:
        """Check if a file or directory should be ignored."""
        if self.ignore_hidden and name.startswith('.'):
            return True
        return any(pattern in name for pattern in self.ignore_patterns)

    def create_file_node(self, file_path: str) -> Dict:
        """Create a node representing a file."""
        stats = os.stat(file_path)
        node = {
            'type': 'file',
            'name': os.path.basename(file_path),
            'path': file_path,
            'size': stats.st_size,
            'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
        }
    
        if self.include_content:
            file_ext = os.path.splitext(file_path)[1].lower()  # Get the file extension
            if self.content_extensions is None or file_ext in self.content_extensions:
                print(f"Reading content for: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        node['content'] = content
                        print(f"Successfully read {len(content)} bytes from {file_path}")
                except Exception as e:
                    print(f"Warning: Could not read content of {file_path}: {e}")
            else:
                print(f"Skipping content for {file_path} - extension not in allowed list")
    
        return node

    def map_directory(self, dir_path: str, depth: int = 0) -> Dict:
        """Recursively map a directory structure."""
        if depth >= self.max_depth:
            return {
                'type': 'directory',
                'name': os.path.basename(dir_path),
                'path': dir_path,
                'children': {'[MAX_DEPTH_REACHED]': {'type': 'file', 'name': '', 'path': ''}}
            }

        result = {
            'type': 'directory',
            'name': os.path.basename(dir_path),
            'path': dir_path,
            'children': {}
        }

        try:
            for entry in os.listdir(dir_path):
                if self.should_ignore(entry):
                    continue

                full_path = os.path.join(dir_path, entry)
                if os.path.isdir(full_path):
                    result['children'][entry] = self.map_directory(full_path, depth + 1)
                else:
                    result['children'][entry] = self.create_file_node(full_path)
        except Exception as e:
            print(f"Error mapping directory {dir_path}: {e}")

        return result

    def generate_markdown(self, node: Dict, level: int = 0) -> str:
        """Generate a markdown representation of the directory structure with file contents."""
        indent = '  ' * level
        output = ''

        if node['type'] == 'directory':
            output += f"{indent}- 📁 **{node['name']}/**\n"
            if 'children' in node:
                for child in sorted(node['children'].values(), key=lambda x: x['name']):
                    output += self.generate_markdown(child, level + 1)
        else:
            output += f"{indent}- 📄 **{node['name']}**\n"
            if self.include_content and 'content' in node:
                file_path = node['path']
                content = node['content']
                # Escape backticks in content
                content = content.replace('```', '```')
                output += f"\n{indent}  📄 *File Path*: `{file_path}`\n\n"
                output += f"{indent}  ```\n"
                output += f"{indent}  {content}\n"
                output += f"{indent}  ```\n\n"

        return output

    def save_map(self, node: Dict, output_path: str, file_format: str = 'json'):
        """Save the directory map to a file."""
        if file_format == 'markdown':
            content = "# Directory Structure\n\n" + self.generate_markdown(node)
        else:
            content = json.dumps(node, indent=2)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


class WeaviateDocCollector:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited = set()
        self.markdown_content = []

    def fetch_page(self, url):
        try:
            response = httpx.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            print(f"Failed to fetch {url}: {e}")
            return None

    def parse_page(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find('title').get_text() if soup.find('title') else 'No Title'
        # Attempt to extract main content from common tags
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        body = main_content.get_text() if main_content else 'No Content'
        print(f"Extracted content from: {title}")  # Log the title of the page being processed
        return f"# {title}\n\n{body}\n\n"

    def collect_docs(self, path="/"):
        full_url = self.base_url + path
        if full_url in self.visited:
            return
        self.visited.add(full_url)

        html_content = self.fetch_page(full_url)
        if html_content:
            page_markdown = self.parse_page(html_content)
            self.markdown_content.append(page_markdown)

            # Example of finding links to other pages
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/') and href not in self.visited:
                    print(f"Following link: {href}")  # Log the link being followed
                    self.collect_docs(href)

    def save_to_markdown(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.markdown_content))


def main():
    parser = argparse.ArgumentParser(description='Map a directory structure or collect Weaviate documentation')
    parser.add_argument('directory', nargs='?', help='Directory to map')
    parser.add_argument('output', nargs='?', help='Output file to save the map')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json', help='Output format')
    parser.add_argument('--max-depth', type=int, default=10, help='Maximum depth to traverse')
    parser.add_argument('--include-content', action='store_true', help='Include file content in the map')
    parser.add_argument('--no-ignore-hidden', action='store_false', dest='ignore_hidden', help='Include hidden files and directories')
    parser.add_argument('--content-extensions', nargs='*', help='List of file extensions to include content')
    args = parser.parse_args()

    if args.directory and args.output:
        mapper = DirectoryMapper(
            max_depth=args.max_depth,
            include_content=args.include_content,
            content_extensions=args.content_extensions,
            ignore_hidden=args.ignore_hidden
        )
        dir_map = mapper.map_directory(args.directory)
        mapper.save_map(dir_map, args.output, args.format)
        print(f"\nMap saved to: {args.output}")
    else:
        base_url = "https://it7yckdquop1dskvjtdna.c0.us-west3.gcp.weaviate.cloud/v1/docs"
        collector = WeaviateDocCollector(base_url)
        collector.collect_docs()
        collector.save_to_markdown("weaviate_docs.md")

if __name__ == "__main__":
    main()