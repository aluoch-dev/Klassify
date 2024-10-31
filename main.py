from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.filechooser import FileChooserIconView
from PIL import Image as PILImage
import numpy as np

# Load the .kv file
Builder.load_file("main.kv")

class ImageClassifierApp(App):
    selected_image_path = ObjectProperty(None)

    def build(self):
        return self.root

    def select_image(self):
        """Open a file chooser to select an image."""
        filechooser = FileChooserIconView(on_selection=lambda instance, value: self.load_image(value[0]))
        self.root.add_widget(filechooser)

    def load_image(self, path):
        """Load and display the selected image."""
        self.selected_image_path = path
        self.root.ids.image_display.source = path
        self.root.ids.result_label.text = "Image selected: " + path.split('/')[-1]

    def classify_image(self):
        """Classify the selected image and display the result."""
        if not self.selected_image_path:
            self.root.ids.result_label.text = "Please select an image first."
            return

        # Load image and preprocess it for the model
        image = PILImage.open(self.selected_image_path).resize((224, 224))  # Adjust size for model's requirement
        image_array = np.array(image) / 255.0  # Normalize if needed
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Dummy classification (replace with your model's prediction code)
        classification_result = self.predict_image(image_array)

        # Display the result
        self.root.ids.result_label.text = f"Classification Result: {classification_result}"

    def predict_image(self, image_array):
        """Dummy prediction function, replace with actual model prediction."""
        return "Classified as Category X"

if __name__ == "__main__":
    ImageClassifierApp().run()
