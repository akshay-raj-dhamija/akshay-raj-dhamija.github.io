# Blog Directory Structure

This directory contains the blog section of the website using **Markdown** for easy content creation with **Jekyll**.

## Structure

```
blog/
├── index.html                    # Main blog landing page
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

Blog posts are written in **Markdown** (`.md` files) and automatically converted to HTML by Jekyll using the `post` layout (`_layouts/post.html`).

### Creating a New Blog Post

1. **Create a Markdown file** in the appropriate category directory:
   ```bash
   touch blog/computer-vision/my-new-post.md
   ```

2. **Add Jekyll front matter** at the top of the file:
   ```markdown
   ---
   layout: post
   title: "Your Post Title"
   permalink: /blog/computer-vision/my-new-post.html
   ---
   ```

3. **Write your content** using standard Markdown syntax (see below for features)

4. **Update `blog/index.html`** to add a link to your post:
   ```html
   <article>
       <a href="computer-vision/my-new-post.html" class="image">
           <img src="Images/your-image.png" alt="" />
       </a>
       <div class="inner">
           <h4><a href="computer-vision/my-new-post.html">Your Post Title</a></h4>
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
- **Images:** `![alt text](../Images/image.png)` (use `../Images/` from subdirectories)
- **Code blocks with syntax highlighting:**
  ````markdown
  ```python
  def hello():
      print("Hello, World!")
  ```
  ````
- **Inline code:** `` `code` ``
- **LaTeX Math:**
  - Inline: `$\(E = mc^2\)$` (use `$\(...\)$` format)
  - Display: `$$E = mc^2$$`
  - Note: Escape asterisks in math: `$\(C^\*\)$` instead of `$\(C^*\)$`
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
---
layout: post
title: "Your Post Title"
permalink: /blog/computer-vision/your-post.html
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
```

## Technical Details

- **Markdown Processor:** [kramdown](https://kramdown.gettalong.org/) (via Jekyll)
- **Syntax Highlighting:** [Rouge](https://github.com/rouge-ruby/rouge) (via Jekyll)
- **LaTeX Rendering:** [MathJax](https://www.mathjax.org/) v3
- **Rendering:** Server-side via Jekyll (build-time conversion)
- **Layout:** `_layouts/post.html` (includes sidebar, navigation, and styling)
- **Images:** Reference from `../Images/` relative to the blog subdirectory (e.g., `blog/deep-learning-frameworks/`)

## Benefits of Markdown + Jekyll

- ✅ **Easy to write:** Simple, readable syntax
- ✅ **Version control friendly:** Plain text files work great with Git
- ✅ **Code highlighting:** Automatic syntax highlighting for code blocks
- ✅ **No HTML knowledge needed:** Focus on content, not markup
- ✅ **Fast loading:** Pre-rendered HTML (better SEO and performance)
- ✅ **Automatic builds:** GitHub Pages builds from Markdown automatically

## Notes

- Blog-specific images should be placed in `blog/Images/` and referenced as `../Images/` from markdown files in subdirectories
- The `_layouts/post.html` template handles all styling and rendering automatically
- Syntax highlighting supports 100+ languages including Python, JavaScript, C++, etc.
- All blog posts are written in Markdown for easy maintenance and version control
- Math equations use `$\(...\)$` format for inline math and `$$...$$` for display math
- Remember to escape asterisks in math expressions: `C^\*` instead of `C^*`
