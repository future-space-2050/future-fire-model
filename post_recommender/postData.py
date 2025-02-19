class PostData:
    def __init__(self, post_data):
        self.post_data = post_data
        self.post_id = None
        self.post_content = None
        self.category = None
        
    def _process_post(self):
        """Extract post_id, post_content, and category from JSON post data."""
        try:
            self.post_id = self.post_data["post_id"]
            self.post_content = self.post_content
            self.category = self.category
        except KeyError as e:
            raise ValueError(f"Post data must contain post_id, content, and category: {e}")
        
    def to_dict(self):
        """Convert post data to a dictionary."""
        self.post_id = self.post_data["post_id"]
        self._detect_tags()
        self._process_post()
        return {
            "post_id": self.post_id,
            "content": self.post_content,
            "category": self.category
        }
        
    def _detect_tags(self):
        """Extract relevant tags from post content."""
        components = self.post_data["content"].split('#')
        content = components[0]
        tags = components[1:] if len(components) > 1 else "Unknown"
        self.post_content = content
        self.category = "".join(tags)
               