# Jekyll plugin to fix image paths in generated HTML
# Converts absolute paths like /blog/Images/... to relative paths

Jekyll::Hooks.register :documents, :post_render do |document|
  # Process documents with 'post' layout
  if document.data['layout'] == 'post'
    # Calculate relative path depth based on URL
    # URL format: /blog/subdir/file.html
    # We need to go up (number of directories - 1) levels to reach blog/, then Images/
    url = document.url || ''
    url_parts = url.split('/').reject(&:empty?)
    # Remove 'blog' and filename, count remaining directories
    directories = url_parts.reject { |p| p == 'blog' || p.end_with?('.html') }
    depth = directories.size  # Number of subdirectories under blog/
    
    if depth > 0
      # Build relative path prefix
      relative_prefix = '../' * depth
      
      # Fix absolute image paths to relative paths
      document.output = document.output.gsub(
        /src="\/blog\/Images\/([^"]+)"/,
        "src=\"#{relative_prefix}Images/\\1\""
      )
    else
      # If in blog/ root, use ../Images/
      document.output = document.output.gsub(
        /src="\/blog\/Images\/([^"]+)"/,
        "src=\"../Images/\\1\""
      )
    end
  end
end

Jekyll::Hooks.register :pages, :post_render do |page|
  # Process pages with 'post' layout
  if page.data['layout'] == 'post'
    url = page.url || ''
    url_parts = url.split('/').reject(&:empty?)
    directories = url_parts.reject { |p| p == 'blog' || p.end_with?('.html') }
    depth = directories.size
    
    if depth > 0
      relative_prefix = '../' * depth
      page.output = page.output.gsub(
        /src="\/blog\/Images\/([^"]+)"/,
        "src=\"#{relative_prefix}Images/\\1\""
      )
    else
      page.output = page.output.gsub(
        /src="\/blog\/Images\/([^"]+)"/,
        "src=\"../Images/\\1\""
      )
    end
  end
end
