#!/usr/bin/env python3
"""
Simple script to convert markdown blog posts to HTML using Jekyll layout
This is a basic preview tool for local testing.

NOTE: For proper LaTeX math rendering, use Jekyll locally:
  bundle install
  bundle exec jekyll serve

Python's markdown library doesn't fully preserve LaTeX math expressions,
so math may not render correctly with this script. Jekyll/kramdown on
GitHub Pages will handle math correctly.
"""

import os
import re
from pathlib import Path

# Blog markdown files to convert
BLOG_FILES = [
    'blog/deep-learning-frameworks/search-algorithms-part1.md',
    'blog/deep-learning-frameworks/search-algorithms-part2.md',
    'blog/deep-learning-frameworks/search-algorithms-part3.md',
    'blog/deep-learning-frameworks/search-trees-to-reinforcement-learning.md',
    'blog/deep-learning-frameworks/caffe-custom-layer.md',
    'blog/deep-learning-frameworks/mnist-tutorial.md',
    'blog/computer-vision/understanding-open-set-recognition.md',
    'blog/life-skills/navigating-phd-journey.md',
]

def read_layout():
    """Read the Jekyll layout file"""
    with open('_layouts/post.html', 'r', encoding='utf-8') as f:
        return f.read()

def extract_front_matter(md_content):
    """Extract front matter from markdown file"""
    if not md_content.startswith('---'):
        return {}, md_content
    
    parts = md_content.split('---', 2)
    if len(parts) < 3:
        return {}, md_content
    
    front_matter_text = parts[1].strip()
    content = parts[2]
    
    # Parse simple front matter (title, layout, permalink)
    front_matter = {}
    for line in front_matter_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            front_matter[key] = value
    
    return front_matter, content

def convert_markdown_to_html(md_content, md_file_path):
    """Convert markdown to HTML"""
    try:
        import markdown
        
        # Use extensions for code highlighting, tables, fenced code blocks
        md = markdown.Markdown(extensions=[
            'codehilite',
            'fenced_code', 
            'tables',
            'nl2br',
            'attr_list'
        ])
        html = md.convert(md_content)
        
        # Note: Python markdown doesn't preserve LaTeX math delimiters perfectly.
        # Math expressions may not render correctly with this script.
        # For proper math rendering, use Jekyll (which handles math natively via kramdown).
        
        # Fix image paths based on file location
        # From blog/subdir/file.md, Images/ should be ../Images/ (to go from blog/subdir to blog/Images/)
        # Count directory depth: blog/subdir/file.md has 2 slashes, so depth = 1 (one subdirectory)
        depth = md_file_path.count('/') - 1  # blog/subdir/file.md = 1 subdirectory
        if depth > 0:
            image_prefix = '../' * depth + 'Images/'
        else:
            image_prefix = 'Images/'
        
        # Replace Images/ with correct relative path
        html = re.sub(r'src="Images/', f'src="{image_prefix}', html)
        html = re.sub(r'href="Images/', f'href="{image_prefix}', html)
        
        # Preserve LaTeX math - ensure \( and \) are not escaped
        # Markdown might escape backslashes, so we fix them
        html = html.replace('\\\\(', '\\(')
        html = html.replace('\\\\)', '\\)')
        html = html.replace('\\\\[', '\\[')
        html = html.replace('\\\\]', '\\]')
        return html
    except ImportError:
        # Fallback: very basic markdown conversion
        html = md_content
        # Headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        # Images
        html = re.sub(r'!\[(.+?)\]\((.+?)\)', r'<img src="\2" alt="\1">', html)
        # Paragraphs (simple)
        html = re.sub(r'\n\n', r'</p><p>', html)
        html = '<p>' + html + '</p>'
        return html

def build_blog_post(md_file, layout_template):
    """Build a single blog post HTML file"""
    print(f"Processing {md_file}...")
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    front_matter, content = extract_front_matter(md_content)
    
    # Get permalink or generate from filename
    if 'permalink' in front_matter:
        html_path = front_matter['permalink'].lstrip('/')
    else:
        html_path = md_file.replace('.md', '.html')
    
    # Convert markdown to HTML
    html_content = convert_markdown_to_html(content, md_file)
    
    # Replace {{ content }} in layout
    html_output = layout_template.replace('{{ content }}', html_content)
    
    # Replace title if present
    if 'title' in front_matter:
        html_output = html_output.replace('{% if page.title %}{{ page.title }} - {% endif %}', 
                                         f"{front_matter['title']} - ")
    
    # Fix asset paths based on file location
    # From blog/subdir/file.html, need to go up (depth+1) levels to reach root
    depth = md_file.count('/') - 1  # blog/subdir/file.md = 2 levels deep (blog + subdir)
    if depth > 0:
        root_prefix = '../' * (depth + 1)  # +1 to go from blog/subdir to root
        blog_prefix = '../' * depth  # to go from blog/subdir to blog
        
        # Replace paths in layout template
        html_output = html_output.replace('../assets/', root_prefix + 'assets/')
        html_output = html_output.replace('../Images/ARD.png', root_prefix + 'Images/ARD.png')
        html_output = html_output.replace('../CV_', root_prefix + 'CV_')
        html_output = html_output.replace('../vr.html', root_prefix + 'vr.html')
        # Fix index.html links - need to handle both with and without anchors
        # Replace ../index.html with blog_prefix + index.html (blog_prefix already includes trailing /)
        html_output = html_output.replace('../index.html', blog_prefix + 'index.html')
        html_output = html_output.replace('href="index.html"', f'href="{blog_prefix}index.html"')
        html_output = html_output.replace('href="index.html#', f'href="{blog_prefix}index.html#')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"  Created {html_path}")

def main():
    print("Building blog posts...")
    layout_template = read_layout()
    
    for md_file in BLOG_FILES:
        if os.path.exists(md_file):
            build_blog_post(md_file, layout_template)
        else:
            print(f"  Warning: {md_file} not found")
    
    print("\nDone! Blog posts built. You can now serve with: python3 -m http.server 8000")

if __name__ == '__main__':
    main()
