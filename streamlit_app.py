import streamlit as st
import cv2
import numpy as np
import os
import math
import csv
import pandas as pd
import io
import zipfile

# ------------------------- CORE LOGIC (Modularized from bulk.py) -------------------------

def remove_crosshair(img):
    if img is None or len(img.shape) != 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_combined = cv2.bitwise_or(mask_blue, mask_purple)
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask_combined, kernel, iterations=2)
    inpainted = cv2.inpaint(img, mask_dilated, 3, cv2.INPAINT_TELEA)
    return inpainted

def segment_faz(img_gray, params):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['tophat_kernel'], params['tophat_kernel']))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel_tophat)
    edges = cv2.Canny(tophat, params['canny_t1'], params['canny_t2'])
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['close_kernel'], params['close_kernel']))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    edges_inv = cv2.bitwise_not(edges_closed)
    
    contours, _ = cv2.findContours(edges_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    
    h, w = edges_inv.shape
    x_min = int(w * params['peripheral_frac'])
    x_max = int(w * (1.0 - params['peripheral_frac']))
    y_min = int(h * params['peripheral_frac'])
    y_max = int(h * (1.0 - params['peripheral_frac']))

    def is_inside_central_region(contour):
        M = cv2.moments(contour)
        if M.get("m00", 0) == 0: return False
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)

    central_contours = [c for c in contours if is_inside_central_region(c)]
    if not central_contours:
        central_contours = contours

    faz_contour = max(central_contours, key=cv2.contourArea)
    mask = np.zeros_like(edges_inv)
    cv2.drawContours(mask, [faz_contour], -1, 255, -1)
    return faz_contour, mask

def compute_metrics(contour, px_to_mm):
    if contour is None: return {}
    
    area_px = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, True))
    
    # Basic Metrics
    area_mm2 = area_px * (px_to_mm**2)
    perimeter_mm = perimeter_px * px_to_mm
    circularity = (4.0 * math.pi * area_px) / (perimeter_px * perimeter_px) if perimeter_px > 0 else 0
    form_factor = circularity
    
    # Convex Hull
    hull = cv2.convexHull(contour)
    hull_area_px = float(cv2.contourArea(hull))
    hull_perim_px = float(cv2.arcLength(hull, True))
    solidity = area_px / hull_area_px if hull_area_px > 0 else 0
    convexity = (hull_perim_px / perimeter_px) if perimeter_px > 0 else 0
    hull_area_mm2 = hull_area_px * (px_to_mm**2)
    
    # Feret
    pts = contour[:, 0, :].astype(np.float64)
    max_dist = 0.0
    pt1_max = pt2_max = None
    if len(pts) > 0:
        for i in range(len(pts)):
            diffs = pts[i+1:] - pts[i]
            dists = np.hypot(diffs[:,0], diffs[:,1])
            if dists.size > 0:
                j = np.argmax(dists)
                if dists[j] > max_dist:
                    max_dist = dists[j]
                    pt1_max, pt2_max = pts[i], pts[i+1+j]

    # Min Feret (Simplified: perp to max)
    min_dist = 0.0
    if pt1_max is not None and pt2_max is not None:
        major_vec = pt2_max - pt1_max
        norm = np.linalg.norm(major_vec)
        if norm > 0:
            perp_unit = np.array([-major_vec[1], major_vec[0]]) / norm
            perp_proj = pts.dot(perp_unit)
            min_dist = float(np.max(perp_proj) - np.min(perp_proj))

    # Ellipse
    ellipse_data = {
        "major_mm": 0.0, "minor_mm": 0.0, "ellipticity": 0.0, 
        "axis_ratio": 0.0, "ovality_index": 0.0, "area_mm2": 0.0, "angle": 0.0, "raw": None
    }
    if len(contour) >= 5:
        try:
            ell = cv2.fitEllipse(contour)
            (cx, cy), (ma, MA), angle = ell
            major = max(ma, MA)
            minor = min(ma, MA)
            a, b = major/2.0, minor/2.0
            ellipse_data = {
                "major_mm": major * px_to_mm,
                "minor_mm": minor * px_to_mm,
                "ellipticity": math.sqrt(max(0.0, 1.0 - (b*b)/(a*a))) if a > 0 else 0,
                "axis_ratio": (b/a) if a > 0 else 0,
                "ovality_index": (major/minor) if minor > 0 else 0,
                "area_mm2": (math.pi * a * b) * (px_to_mm**2),
                "angle": angle,
                "raw": ell
            }
        except: pass

    return {
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter_mm,
        "circularity": circularity,
        "form_factor": form_factor,
        "feret_max_mm": max_dist * px_to_mm,
        "feret_min_mm": min_dist * px_to_mm,
        "ellipse_major_mm": ellipse_data["major_mm"],
        "ellipse_minor_mm": ellipse_data["minor_mm"],
        "ellipse_ellipticity": ellipse_data["ellipticity"],
        "ellipse_axis_ratio": ellipse_data["axis_ratio"],
        "ellipse_ovality_index": ellipse_data["ovality_index"],
        "ellipse_area_mm2": ellipse_data["area_mm2"],
        "ellipse_angle_deg": ellipse_data["angle"],
        "solidity": solidity,
        "convexity": convexity,
        "hull_area_mm2": hull_area_mm2,
        "ellipse_raw": ellipse_data["raw"]
    }

# ------------------------- STREAMLIT UI -------------------------

st.set_page_config(page_title="FAZ Analysis Pro", layout="wide")

# Session State Initialization
if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = None
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'image_cache' not in st.session_state:
    st.session_state.image_cache = {}

st.title("👁️ FAZ Analysis by Chamika Madushanka")

# Sidebar - Parameters
st.sidebar.header("Global Analysis Settings")
scan_size = st.sidebar.number_input("Scan Size (mm)", value=3.0)
image_pixels = st.sidebar.number_input("Image Pixels (px)", value=450)
px_to_mm = scan_size / image_pixels

st.sidebar.markdown("---")
st.sidebar.subheader("Segmentation Tuning")
tophat_k = st.sidebar.slider("Top-hat Kernel Size", 4, 30, 10, key="global_tophat")
canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 100, 0, key="global_t1")
canny_t2 = st.sidebar.slider("Canny Threshold 2", 100, 500, 250, key="global_t2")
close_k = st.sidebar.slider("Closing Kernel Size", 10, 80, 40, key="global_close")
periph_frac = st.sidebar.slider("Peripheral Exclusion (%)", 0, 40, 20, key="global_periph") / 100.0

params = {
    'tophat_kernel': tophat_k,
    'canny_t1': canny_t1,
    'canny_t2': canny_t2,
    'close_kernel': close_k,
    'peripheral_frac': periph_frac
}

tabs = st.tabs(["📸 Single Image Analysis", "📂 Bulk Processing"])

# --- TAB 1: SINGLE IMAGE ---
with tabs[0]:
    uploaded_file = st.file_uploader("Choose an OCTA image...", type=["png", "jpg", "jpeg", "tif"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process
        img_clean = remove_crosshair(img)
        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
        contour, mask = segment_faz(gray, params)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(img, channels="BGR", use_container_width=True)
            
        with col2:
            st.subheader("Real-time FAZ Mask")
            if contour is not None:
                vis = img_clean.copy()
                cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
                metrics = compute_metrics(contour, px_to_mm)
                
                # Draw ellipse if available
                if metrics.get("ellipse_raw"):
                    cv2.ellipse(vis, metrics["ellipse_raw"], (0, 255, 255), 2)
                
                st.image(vis, channels="BGR", use_container_width=True)
                
                # Display Summary Metrics
                st.markdown("### 📊 Metrics Summary")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Area", f"{metrics['area_mm2']:.4f} mm²")
                m_col2.metric("Perimeter", f"{metrics['perimeter_mm']:.4f} mm")
                m_col3.metric("Circularity", f"{metrics['circularity']:.4f}")
                
                # Detailed Metrics Dropdown
                with st.expander("🔍 View All 16 Detailed Metrics"):
                    st.write("---")
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.write(f"**Area:** {metrics['area_mm2']:.6f} mm²")
                        st.write(f"**Perimeter:** {metrics['perimeter_mm']:.6f} mm")
                        st.write(f"**Circularity:** {metrics['circularity']:.6f}")
                        st.write(f"**Form Factor:** {metrics['form_factor']:.6f}")
                        st.write(f"**Feret Max:** {metrics['feret_max_mm']:.6f} mm")
                        st.write(f"**Feret Min:** {metrics['feret_min_mm']:.6f} mm")
                        st.write(f"**Solidity:** {metrics['solidity']:.6f}")
                        st.write(f"**Convexity:** {metrics['convexity']:.6f}")
                    with d_col2:
                        st.write(f"**Ellipse Major:** {metrics['ellipse_major_mm']:.6f} mm")
                        st.write(f"**Ellipse Minor:** {metrics['ellipse_minor_mm']:.6f} mm")
                        st.write(f"**Ellipticity:** {metrics['ellipse_ellipticity']:.6f}")
                        st.write(f"**Axis Ratio:** {metrics['ellipse_axis_ratio']:.6f}")
                        st.write(f"**Ovality Index:** {metrics['ellipse_ovality_index']:.6f}")
                        st.write(f"**Ellipse Area:** {metrics['ellipse_area_mm2']:.6f} mm²")
                        st.write(f"**Ellipse Angle:** {metrics['ellipse_angle_deg']:.2f}°")
                        st.write(f"**Hull Area:** {metrics['hull_area_mm2']:.6f} mm²")
            else:
                st.warning("Could not detect FAZ with current settings.")

# --- TAB 2: BULK PROCESSING ---
with tabs[1]:
    st.subheader("Batch Analysis Control")
    uploaded_files = st.file_uploader("Upload multiple OCTA images...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    
    if st.button("🚀 Run Analysis on Uploaded Files", use_container_width=True) and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_list = []
        st.session_state.image_cache = {}
        
        for idx, up_file in enumerate(uploaded_files):
            fname = up_file.name
            status_text.text(f"Processing: {fname}")
            
            try:
                # Read file bytes and decode to OpenCV image
                file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.image_cache[fname] = img # Cache original for re-tuning
                
                img_clean = remove_crosshair(img)
                gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
                contour, mask = segment_faz(gray, params)
                
                if contour is not None:
                    metrics = compute_metrics(contour, px_to_mm)
                    metrics['FileName'] = fname
                    results_list.append(metrics)
            except Exception as e:
                st.error(f"Error processing {fname}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if results_list:
            st.session_state.bulk_results = pd.DataFrame(results_list)
            st.success(f"✅ Processed {len(results_list)} images.")

    # Results Inspection Area
    if st.session_state.bulk_results is not None:
        st.markdown("---")
        
        # ZIP Preparation
        df = st.session_state.bulk_results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add CSV
            csv_data = df.to_csv(index=False).encode('utf-8')
            zip_file.writestr("faz_results.csv", csv_data)
            
            # Add images
            for fname, img in st.session_state.image_cache.items():
                img_clean = remove_crosshair(img)
                gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
                c, m = segment_faz(gray, params)
                if c is not None:
                    vis = img_clean.copy()
                    cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
                    # Encode to PNG in memory
                    _, buffer = cv2.imencode('.png', vis)
                    zip_file.writestr(f"masked_{fname}", buffer.tobytes())

        st.download_button(
            label="📂 Download Results as ZIP (CSV + Masks)",
            data=zip_buffer.getvalue(),
            file_name="faz_analysis_results.zip",
            mime="application/zip",
            use_container_width=True
        )

        st.subheader("🔍 Interactive Results Inspection")
        img_names = df['FileName'].tolist()
        selected_name = st.selectbox("Select a processed image to review/re-tune:", img_names)
        
        if selected_name:
            st.session_state.selected_image = selected_name
            row = df[df['FileName'] == selected_name].iloc[0]
            
            # Re-tune UI for selected image
            i_col1, i_col2 = st.columns([1, 2])
            
            with i_col1:
                st.markdown("#### Local Adjustments")
                t_k = st.slider("Top-hat Kernel", 4, 30, params['tophat_kernel'], key="local_tophat")
                t1 = st.slider("Canny T1", 0, 100, params['canny_t1'], key="local_t1")
                t2 = st.slider("Canny T2", 100, 500, params['canny_t2'], key="local_t2")
                c_k = st.slider("Close Kernel", 10, 80, params['close_kernel'], key="local_close")
                p_f = st.slider("Peripheral (%)", 0, 40, int(params['peripheral_frac']*100), key="local_periph") / 100.0
                
                local_params = {
                    'tophat_kernel': t_k, 'canny_t1': t1, 'canny_t2': t2,
                    'close_kernel': c_k, 'peripheral_frac': p_f
                }
                
                if st.button("Apply and Update Bulk Results"):
                    orig = st.session_state.image_cache.get(selected_name)
                    if orig is not None:
                        clean = remove_crosshair(orig)
                        g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
                        c, m = segment_faz(g, local_params)
                        if c is not None:
                            new_metrics = compute_metrics(c, px_to_mm)
                            new_metrics['FileName'] = selected_name
                            
                            # Update dataframe
                            idx = df[df['FileName'] == selected_name].index[0]
                            for k, v in new_metrics.items():
                                df.at[idx, k] = v
                            st.session_state.bulk_results = df
                            st.rerun()

            with i_col2:
                # Preview
                orig = st.session_state.image_cache.get(selected_name)
                if orig is not None:
                    clean = remove_crosshair(orig)
                    g = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
                    c, m = segment_faz(g, local_params)
                    
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        st.image(orig, channels="BGR", caption="Original", use_container_width=True)
                    with p_col2:
                        if c is not None:
                            v = clean.copy()
                            cv2.drawContours(v, [c], -1, (0, 255, 0), 2)
                            metrics_local = compute_metrics(c, px_to_mm)
                            if metrics_local.get("ellipse_raw"):
                                cv2.ellipse(v, metrics_local["ellipse_raw"], (0, 255, 255), 2)
                            st.image(v, channels="BGR", caption="Processed Mask", use_container_width=True)
                        else:
                            st.warning("FAZ not found with these settings.")
            
            # Display detailed metrics for the selected selection
            with st.expander(f"📊 Detailed Metrics for {selected_name}"):
                metrics = row.to_dict()
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write(f"**Area:** {metrics['area_mm2']:.6f} mm²")
                    st.write(f"**Perimeter:** {metrics['perimeter_mm']:.6f} mm")
                    st.write(f"**Circularity:** {metrics['circularity']:.6f}")
                    st.write(f"**Form Factor:** {metrics['form_factor']:.6f}")
                    st.write(f"**Feret Max:** {metrics['feret_max_mm']:.6f} mm")
                    st.write(f"**Feret Min:** {metrics['feret_min_mm']:.6f} mm")
                    st.write(f"**Solidity:** {metrics['solidity']:.6f}")
                    st.write(f"**Convexity:** {metrics['convexity']:.6f}")
                with d_col2:
                    st.write(f"**Ellipse Major:** {metrics['ellipse_major_mm']:.6f} mm")
                    st.write(f"**Ellipse Minor:** {metrics['ellipse_minor_mm']:.6f} mm")
                    st.write(f"**Ellipticity:** {metrics['ellipse_ellipticity']:.6f}")
                    st.write(f"**Axis Ratio:** {metrics['ellipse_axis_ratio']:.6f}")
                    st.write(f"**Ovality Index:** {metrics['ellipse_ovality_index']:.6f}")
                    st.write(f"**Ellipse Area:** {metrics['ellipse_area_mm2']:.6f} mm²")
                    st.write(f"**Ellipse Angle:** {metrics['ellipse_angle_deg']:.2f}°")
                    st.write(f"**Hull Area:** {metrics['hull_area_mm2']:.6f} mm²")

        st.markdown("### 📋 Final Bulk Data Table")
        st.dataframe(st.session_state.bulk_results.drop(columns=['ellipse_raw'], errors='ignore'))
