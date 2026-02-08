import streamlit as st
import cv2
import numpy as np
import pandas as pd
import svgwrite
import cairosvg
import io
from pdf2image import convert_from_bytes
from PIL import Image

st.set_page_config(page_title="EdTech OMR - V2", layout="wide")

# --- 1. FONCTIONS UTILITAIRES (GÉNÉRATEUR) ---
def generer_svg_omr(titre, sous_titre, nb_questions):
    width = "210mm"
    height = "297mm"
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
    
    # Titres
    dwg.add(dwg.text(titre, insert=("20mm", "20mm"), font_size="20px", font_family="Arial", font_weight="bold"))
    dwg.add(dwg.text(sous_titre, insert=("20mm", "30mm"), font_size="14px", font_family="Arial"))

    # Repères visuels (CADRE NOIR AUTOUR DE LA ZONE DE RÉPONSE POUR AIDER LA DÉTECTION)
    # On dessine un cadre rectangulaire épais qui servira de repère
    dwg.add(dwg.rect(insert=("15mm", "40mm"), size=("180mm", "230mm"), fill="none", stroke="black", stroke_width="5"))

    # Grille de questions
    x_start_col1 = 30
    x_start_col2 = 110
    y_start = 50
    y_gap = 10
    
    for i in range(1, nb_questions + 1):
        if i <= 20:
            x_pos = x_start_col1
            y_pos = y_start + ((i-1) * y_gap) # Ajustement index
        else:
            x_pos = x_start_col2
            y_pos = y_start + ((i - 21) * y_gap)
            
        # Numéro et Bulles
        dwg.add(dwg.text(f"{i}.", insert=(f"{x_pos}mm", f"{y_pos+5}mm"), font_size="12px"))
        options = ['A', 'B', 'C', 'D', 'E']
        for idx, opt in enumerate(options):
            cx = x_pos + 10 + (idx * 8)
            cy = y_pos + 4
            dwg.add(dwg.circle(center=(f"{cx}mm", f"{cy}mm"), r="3mm", stroke="black", fill="white", stroke_width=1))
            dwg.add(dwg.text(opt, insert=(f"{cx-1}mm", f"{cy+1}mm"), font_size="8px", font_family="Arial", fill="gray"))

    return dwg.tostring()

# --- 2. FONCTIONS DE VISION (LE COEUR DU RÉACTEUR) ---

def order_points(pts):
    """Ordonne les coordonnées (Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche)"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Redresse l'image (Perspective Warp)"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def trouver_feuille(image_cv):
    """Trouve le plus grand quadrilatère (la feuille ou le cadre réponse)"""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
            
    return docCnt, edged # On retourne aussi 'edged' pour le debug

def analyser_grille(warped_img, nb_questions=20):
    """Découpe la grille redressée et trouve les réponses"""
    # 1. Binarisation
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # C'est ici que la magie opère : On divise l'image en grille virtuelle
    # Note: Ceci est une approximation. Pour un système robuste, il faut calibrer les coordonnées.
    # Pour l'exemple, on suppose que le 'warped_img' est exactement le cadre noir dessiné par le générateur.
    
    h, w = thresh.shape
    reponses = {}
    
    # On découpe arbitrairement en lignes pour l'exemple
    # (Dans un vrai projet, on utilise les contours des bulles)
    
    # SIMULATION INTELLIGENTE (au lieu du random)
    # On compte les pixels noirs totaux pour voir si la feuille est vide
    total_pixels_noirs = cv2.countNonZero(thresh)
    ratio_remplissage = total_pixels_noirs / (h * w)
    
    status = "Feuille vide détectée"
    if ratio_remplissage > 0.05: # Si plus de 5% de noir
        status = "Réponses détectées"
        # Ici on mettrait la logique complexe de découpage des bulles
        note = "Analyse en cours..." 
    else:
        note = "0/20 (Vide)"

    return thresh, note, status

# --- 3. INTERFACE ---
st.sidebar.title("EdTech OMR V2")
mode = st.sidebar.radio("Mode", ["Générateur", "Correcteur"])

if mode == "Générateur":
    st.title("Générateur de Gabarit")
    if st.button("Générer PDF Test"):
        svg = generer_svg_omr("Test", "Noircir les cases", 20)
        pdf = cairosvg.svg2pdf(bytestring=svg.encode('utf-8'))
        st.download_button("Télécharger PDF", pdf, "gabarit_v2.pdf", "application/pdf")
        st.image(convert_from_bytes(pdf)[0], caption="Aperçu (Cadre ajouté)", width=300)

elif mode == "Correcteur":
    st.title("Correcteur OMR")
    fichiers = st.file_uploader("Scans", accept_multiple_files=True, type=['pdf', 'png', 'jpg'])
    
    if fichiers:
        resultats = []
        for f in fichiers:
            # 1. Conversion sécurisée (Fond blanc)
            bytes_data = f.read()
            if f.type == "application/pdf":
                pil_img = convert_from_bytes(bytes_data)[0].convert('RGB') # Force RGB (Fond blanc)
                img = np.array(pil_img)
                img = img[:, :, ::-1].copy() # RGB -> BGR pour OpenCV
            else:
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)

            # 2. Recherche de la feuille
            docCnt, debug_edge = trouver_feuille(img)
            
            if docCnt is not None:
                # Dessin du contour trouvé en ROUGE sur l'image originale
                debug_vis = img.copy()
                cv2.drawContours(debug_vis, [docCnt], -1, (0, 0, 255), 10) # Gros trait rouge
                
                # Redressement
                warped = four_point_transform(img, docCnt.reshape(4, 2))
                
                # Analyse
                img_analyse, note, status = analyser_grille(warped)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(debug_vis, cv2.COLOR_BGR2RGB), caption="Détection (Cadre Rouge)", width=300)
                with col2:
                    st.image(img_analyse, caption="Vision Ordinateur (Binarisé)", width=300)
                
                resultats.append({"Fichier": f.name, "Note": note, "État": status})
            else:
                st.error(f"Impossible de trouver le cadre sur {f.name}")
                st.image(debug_edge, caption="Ce que voit l'ordi (Edges)", width=300)
                resultats.append({"Fichier": f.name, "Note": "Erreur", "État": "Cadre non trouvé"})

        st.dataframe(pd.DataFrame(resultats))
