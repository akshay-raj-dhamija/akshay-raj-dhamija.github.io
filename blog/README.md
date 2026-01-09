# Blog Directory Structure

This directory contains the blog section of the website using **Markdown** for easy content creation.

## Structure

```
blog/
├── index.html                    # Main blog landing page
├── post.html                     # Markdown renderer template
├── Images/                       # Blog-specific images
│   ├── DFS.png
│   ├── BFS.png
│   ├── Alpha_Beta.png
│   └── ... (other blog images)
├── computer-vision/              # Computer Vision & Deep Learning articles
│   └── understanding-open-set-recognition.md
├── deep-learning-frameworks/     # Deep Learning Framework tutorials & AI Fundamentals
│   ├── search-algorithms-part1.md               # Part 1: DFS to A-Star (10 min)
│   ├── search-algorithms-part2.md               # Part 2: Multi-Agent Search (8 min)
│   ├── search-algorithms-part3.md               # Part 3: MDPs to Q-Learning (12 min)
│   ├── search-trees-to-reinforcement-learning.md  # Complete guide (all 3 parts, 35 min)
│   ├── caffe-custom-layer.md
│   └── mnist-tutorial.md
├── life-skills/                  # Life Skills & Career articles
│   └── navigating-phd-journey.md
└── README.md                     # This file
```

## Categories

### Computer Vision & Deep Learning
Articles about research, techniques, and insights in computer vision and machine learning.

### Deep Learning Frameworks & AI Fundamentals
Comprehensive guides covering AI fundamentals (search algorithms, MDPs, reinforcement learning) and practical tutorials for working with Caffe, Caffe2, PyTorch, TensorFlow, and other deep learning frameworks.

### Life Skills & Career
Lessons learned, career advice, and thoughts on personal development.

## Writing Blog Posts in Markdown

Blog posts are now written in **Markdown** (`.md` files) for easier maintenance. The `post.html` template automatically converts Markdown to HTML using [Marked.js](https://marked.js.org/) and adds syntax highlighting with [Highlight.js](https://highlightjs.org/).

### Creating a New Blog Post

1. **Create a Markdown file** in the appropriate category directory:
   ```bash
   touch blog/computer-vision/my-new-post.md
   ```

2. **Write your content** using standard Markdown syntax (see below for features)

3. **Update `blog/index.html`** to add a link to your post:
   ```html
   <article>
       <a href="post.html?md=computer-vision/my-new-post.md" class="image">
           <img src="Images/your-image.png" alt="" />
       </a>
       <div class="inner">
           <h4><a href="post.html?md=computer-vision/my-new-post.md">Your Post Title</a></h4>
           <p>Brief description of your post.</p>
           <p><em>Date</em></p>
       </div>
   </article>
   ```

### Markdown Features Supported

- **Headings:** `# H1`, `## H2`, `### H3`, etc.
- **Bold:** `**bold text**`
- **Italic:** `*italic text*`
- **Links:** `[text](url)`
- **Images:** `![alt text](image-path)`
- **Code blocks with syntax highlighting:**
  ````markdown
  ```python
  def hello():
      print("Hello, World!")
  ```
  ````
- **Inline code:** `` `code` ``
- **LaTeX Math:**
  - Inline: `$E = mc^2$`
  - Display: `$$\sum_{i=1}^{n} x_i$$`
- **Lists:** 
  - Unordered: `- item`
  - Ordered: `1. item`
- **Blockquotes:** `> quote`
- **Horizontal rules:** `---`
- **Tables:**
  ```markdown
  | Header 1 | Header 2 |
  |----------|----------|
  | Cell 1   | Cell 2   |
  ```

### Example Markdown Post Structure

```markdown
# Your Post Title

**Date** | *Category*

[← Back to Blog](../index.html)

---

## Introduction

Your introduction here...

## Code Example

```python
def example():
    return "Hello, World!"
```

## Conclusion

Your conclusion here...

---

[← Back to Blog](../index.html)
```

## Technical Details

- **Markdown Parser:** [Marked.js](https://marked.js.org/) v11.1.1
- **Syntax Highlighting:** [Highlight.js](https://highlightjs.org/) v11.9.0 with Monokai theme
- **LaTeX Rendering:** [MathJax](https://www.mathjax.org/) v3
- **Rendering:** Client-side JavaScript (no build step required)
- **Images:** Reference from `../Images/` relative to the blog subdirectory (e.g., `blog/deep-learning-frameworks/`) or `Images/` relative to the blog directory

## Benefits of Markdown

- ✅ **Easy to write:** Simple, readable syntax
- ✅ **Version control friendly:** Plain text files work great with Git
- ✅ **Code highlighting:** Automatic syntax highlighting for code blocks
- ✅ **No HTML knowledge needed:** Focus on content, not markup
- ✅ **Portable:** Can be converted to other formats if needed

## Notes

- Blog-specific images should be placed in `blog/Images/` and referenced as `../Images/` from markdown files in subdirectories (e.g., `blog/deep-learning-frameworks/`) or `Images/` from `blog/index.html`
- The `post.html` template handles all styling and rendering automatically
- Syntax highlighting supports 100+ languages including Python, JavaScript, C++, etc.
- All blog posts are written in Markdown for easy maintenance and version control

