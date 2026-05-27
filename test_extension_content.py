import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONTENT_JS = PROJECT_ROOT / "extension" / "content.js"


class ExtensionContentTests(unittest.TestCase):
    def setUp(self):
        self.source = CONTENT_JS.read_text()

    def _pdf_icon_markup_source(self):
        match = re.search(
            r"function buildPdfIconMarkup\(\) \{\n(?P<body>.*?)\n\}\n\nfunction findButtonIcon",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        return match.group("body")

    def test_pdf_button_uses_current_youtube_button_shape_classes(self):
        self.assertIn("ytSpecButtonShapeNextHost", self.source)
        self.assertIn("ytSpecButtonShapeNextIcon", self.source)
        self.assertIn("ytSpecButtonShapeNextButtonTextContent", self.source)

    def test_pdf_icon_is_a_filled_youtube_style_glyph(self):
        icon_source = self._pdf_icon_markup_source()

        self.assertIn("ytIconWrapperHost", icon_source)
        self.assertIn("fill: currentcolor", icon_source)
        self.assertNotIn("<circle", icon_source)
        self.assertNotIn("stroke=", icon_source)


if __name__ == "__main__":
    unittest.main()
