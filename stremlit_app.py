# streamlit_app.py
"""
AnonVision - Streamlit GUI Demo
Interactive web interface for testing face blurring system.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

from detector import IntegratedDetector
from decision.decision_module import DecisionModule
from blurring.blurring_module import FaceBlurrer, SelectiveBlurPipeline


# Page config
st.set_page_config(
    page_title="AnonVision - Face Blurring System",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    with st.spinner("Loading AI models... This may take a minute..."):
        st.session_state.detector = IntegratedDetector()
        st.session_state.initialized = True

# Title and description
st.title("🎭 AnonVision - Context-Aware Face Blurring")
st.markdown("""
**Intelligent face detection and selective blurring system**  
Upload an image, adjust settings, and see real-time selective face blurring in action!
""")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")

# Blur settings
st.sidebar.subheader("Blur Configuration")
blur_type = st.sidebar.selectbox(
    "Blur Technique",
    options=['gaussian', 'pixelate', 'mosaic', 'black_box'],
    index=0,
    help="Select the blur effect to apply"
)

blur_intensity = st.sidebar.select_slider(
    "Blur Intensity",
    options=['low', 'medium', 'high'],
    value='medium'
)

# Decision settings
st.sidebar.subheader("Decision Logic")
decision_mode = st.sidebar.selectbox(
    "Decision Mode",
    options=['all', 'largest', 'crowd', 'center_focus'],
    index=0,
    help="""
    - **all**: Blur all detected faces
    - **largest**: Blur all except the largest face
    - **crowd**: Blur only if >3 faces detected
    - **center_focus**: Blur faces far from center
    """
)

user_text = st.sidebar.text_input(
    "User Instruction (Optional)",
    placeholder="e.g., 'blur students', 'blur all'",
    help="Natural language instruction for selective blurring"
)

# Display settings
st.sidebar.subheader("Display Options")
show_debug = st.sidebar.checkbox("Show Debug Info", value=True)
show_original = st.sidebar.checkbox("Show Original Image", value=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📹 Webcam (Coming Soon)", "ℹ️ About"])

with tab1:
    st.header("Upload and Process Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing faces to blur"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            frame = img_array
        
        # Display original
        if show_original:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
        else:
            col2 = st.container()
        
        # Process button
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            with st.spinner("Processing image... Please wait..."):
                try:
                    # Initialize pipeline
                    decision = DecisionModule(mode=decision_mode)
                    blurrer = FaceBlurrer(
                        blur_type=blur_type,
                        blur_intensity=blur_intensity
                    )
                    pipeline = SelectiveBlurPipeline(
                        st.session_state.detector,
                        decision,
                        blurrer
                    )
                    
                    # Process frame
                    blurred_frame, results = pipeline.process_frame(
                        frame,
                        user_text=user_text if user_text else None,
                        show_debug=show_debug
                    )
                    
                    # Convert back to RGB for display
                    blurred_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display result
                    with col2 if show_original else st.container():
                        st.subheader("Processed Image")
                        st.image(blurred_rgb, use_container_width=True)
                    
                    # Display statistics
                    st.success("✅ Processing complete!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Scene Detected", results['scene'])
                    with col_b:
                        st.metric("Persons Detected", len(results['detections']))
                    with col_c:
                        blurred_count = sum(1 for d in results['detections'] if d['bbox_face'])
                        st.metric("Faces Blurred", blurred_count)
                    
                    # Detailed results
                    with st.expander("📊 Detailed Detection Results"):
                        for i, det in enumerate(results['detections'], 1):
                            st.markdown(f"**Person {i}:**")
                            
                            col_x, col_y = st.columns(2)
                            
                            with col_x:
                                if det['attributes']:
                                    st.write(f"- Age: ~{det['attributes']['age']}")
                                    st.write(f"- Gender: {det['attributes']['gender']}")
                                    st.write(f"- Emotion: {det['attributes']['emotion']}")
                            
                            with col_y:
                                if det['dress_color']:
                                    st.write(f"- Clothing: {det['dress_color']}")
                                if det['bbox_face']:
                                    st.write("- Face: ✓ Detected & Blurred")
                                else:
                                    st.write("- Face: ✗ Not detected")
                            
                            st.divider()
                    
                    # Download button
                    result_pil = Image.fromarray(blurred_rgb)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        result_pil.save(tmp.name)
                        tmp_path = tmp.name
                    
                    with open(tmp_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Result",
                            data=f,
                            file_name=f"blurred_{uploaded_file.name}",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    # Cleanup
                    os.unlink(tmp_path)
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👆 Upload an image to get started!")
        
        # Example images showcase
        st.markdown("---")
        st.subheader("📸 Example Use Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Classroom Scenario**")
            st.markdown("- Detect teacher vs students\n- Blur students only\n- Preserve teacher's face")
        
        with col2:
            st.markdown("**Street Photography**")
            st.markdown("- Blur all faces\n- Protect privacy\n- Maintain scene context")
        
        with col3:
            st.markdown("**Group Photos**")
            st.markdown("- Blur background faces\n- Keep main subject clear\n- Smart center focus")

with tab2:
    st.header("Real-time Webcam Processing")
    st.info("🚧 Webcam integration coming soon! Use the command-line tool for real-time processing.")
    st.code("python main.py --mode webcam", language="bash")

with tab3:
    st.header("About AnonVision")
    
    st.markdown("""
    ### 🎯 Features
    
    **Detection Pipeline:**
    - 👤 Person detection (YOLOv8)
    - 😊 Face detection with attributes (age, gender, emotion)
    - 👕 Clothing analysis (type and color)
    - 🤸 Pose estimation (standing, sitting, walking)
    - 🏢 Scene classification (365 categories)
    
    **Intelligent Decision Making:**
    - 🧠 Context-aware blurring rules
    - 📝 Natural language instructions
    - 🎭 Role classification (teacher/student)
    - 🎯 Multiple decision modes
    
    **Blur Techniques:**
    - 🌫️ Gaussian blur (smooth)
    - 🟦 Pixelation (mosaic effect)
    - ⬛ Black box (complete coverage)
    - 🎨 Mosaic (artistic blend)
    
    ### 🛠️ Technology Stack
    
    - **Detection**: YOLOv8, OpenCV, DeepFace, MediaPipe
    - **Scene Understanding**: Places365 (ResNet50)
    - **Clothing Analysis**: OpenAI CLIP
    - **Decision Logic**: Custom ML + Rule-based
    - **NLP**: SentenceTransformers
    - **Interface**: Streamlit, FastAPI
    
    ### 📚 Documentation
    
    See `DETECTION.md` for complete API documentation and usage examples.
    
    ### ⚖️ Privacy & Ethics
    
    This tool is designed for:
    - ✅ Privacy protection in public spaces
    - ✅ Educational content (blur students)
    - ✅ Street photography compliance
    - ✅ Research and development
    
    **Not intended for:**
    - ❌ Surveillance or stalking
    - ❌ Deceptive manipulation
    - ❌ Circumventing consent
    
    ### 📄 License
    
    Integrates multiple open-source models. Check individual licenses:
    - YOLOv8: AGPL-3.0
    - DeepFace: MIT
    - CLIP: MIT
    - MediaPipe: Apache 2.0
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "AnonVision v1.0 | Built with ❤️ using Python, OpenCV, and AI"
    "</div>",
    unsafe_allow_html=True
)