# Autonomous Semantic Segmentation: DINOv2 & SAM Integration

This repository serves as a baseline and educational playground for combining two powerful computer vision foundation models: **DINOv2** (Self-Supervised Vision Transformer) and **SAM** (Segment Anything Model). 

The goal is to provide a clean, ready-to-run environment for team members to experiment with these models, understand their individual capabilities, and determine how they can be integrated into an autonomous pipeline.

## 🧠 Why DINO and SAM?

*   **DINOv2 (The Brain/Eyes):** Developed by Meta, DINOv2 is a Vision Transformer trained without labels. It excels at understanding semantic features and object boundaries natively. It knows *what* and *where* things are, but it doesn't output pixel-perfect masks.
*   **SAM (The Hands):** Also from Meta, SAM is a promptable segmentation model. Given a point or a bounding box, it can extract a perfect mask for that specific object. However, SAM is "blind" to semantics; it doesn't know *what* it is cutting; it just cuts where you tell it to.

**The Integration:** By combining them, we use DINOv2 to automatically find the most semantically important point (salient feature) in an image. We then feed those (x, y) coordinates directly into SAM as a point prompt. The result is a zero-shot, autonomous segmentation pipeline that requires zero human input.

## 📂 Repository Structure

The project is broken down into three step-by-step notebooks:

1.  `dino_exploration.ipynb`: Loads a sample image and extracts the Attention Maps from DINOv2 to visualize how the model "looks" at the image.
2.  `sam_exploration.ipynb`: Demonstrates how SAM works using a manually hardcoded point prompt (e.g., clicking on a cat's face) to generate a mask.
3.  `dino_sam_pipeline.ipynb`: The integration script. It automatically pulls the peak attention point from DINOv2 and passes it to SAM for autonomous mask generation.
4.  `input_images/` & `output_images/`: Directories containing the sample image and the generated visual results.

## 🔍 Case Study: The "Attention Sink" Phenomenon

If you examine the output of `03_autonomous_pipeline.png`, you will notice an interesting artifact:

*   **The Issue:** Instead of targeting the cat (the obvious main subject), DINOv2 placed its highest attention point in the extreme top-left corner of the image. Consequently, SAM perfectly segmented that small piece of the background wall.
*   **The Science Behind It:** This is a known behavior in Vision Transformers (ViTs) called the **"Attention Sink."** Because ViTs process images as sequences of patches, they often use extreme edge patches (like the top-left corner) as a dumping ground to store global context or to discard irrelevant information. The accumulated attention on these "sink" patches can sometimes numerically exceed the attention on the actual object.
*   **The Takeaway:** This pipeline demonstrates that while foundation models are incredibly powerful, understanding their architectural quirks is crucial for building robust AI systems. 

## 💻 Development Environment & AI Tools

This project is designed to be seamlessly executed within our **Jupyter Hub** environment, which comes packed with tools to enhance your workflow:

*   **Jupyter Hub:** You can run the `.ipynb` notebooks directly from the classic Jupyter interface or JupyterLab to visualize outputs step-by-step.
*   **VS Code in Browser:** For a full IDE experience, you can launch **Visual Studio Code** directly from the Jupyter Hub launcher. This opens a fully functional VS Code instance right in your web browser, allowing you to edit scripts, manage files, and use the terminal effortlessly without installing anything on your local machine.
*   **Gemini Code Assistant:** While using the web-based VS Code, you can leverage the built-in **Gemini Code Assistant** extension. This AI tool acts as your pair programmer. You can ask it to explain complex AI architectures, write boilerplate code, debug errors, or suggest refactoring strategies. Just open the Gemini chat panel in your browser's VS Code to start interacting!

## 🛠️ How to Run

1.  Clone this repository to your Jupyter Hub workspace.
2.  Ensure you have a Python virtual environment set up.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Download the SAM ViT-B weights:
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
5.  Run the notebooks in order: 01 -> 02 -> 03.
EOF
