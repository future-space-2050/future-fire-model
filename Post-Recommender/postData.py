class PostData:
    def __init__(self, post_data):
        self.post_data = post_data
        self.post_id = None
        self.post_content = None
        self.category = None
        
    def process_post(self):
        """Extract post_id, post_content, and category from JSON post data."""
        try:
            self.post_id = self.post_data["post_id"]
            self.post_content = self.post_data["content"]
            self.category = self.post_data["category"]
        except KeyError as e:
            raise ValueError(f"Post data must contain post_id, content, and category: {e}")
        
    def to_dict(self):
        """Convert post data to a dictionary."""
        self.post_id = self.post_data["post_id"]
        self.detect_tags()
        return {
            "post_id": self.post_id,
            "content": self.post_content,
            "category": self.category
        }
        
    def detect_tags(self):
        """Extract relevant tags from post content."""
        content, tags = self.post_data["content"].split('#')
        self.post_content = content
        self.category = " ".join(tags)
               