import streamlit as st
import cv2
import numpy as np
import pandas as pd
import svgwrite
import cairosvg
from pdf2image import convert_from_bytes
from PIL import Image

st.set_page_config(page_title="EdTech OMR - V3", layout="wide", page_icon="📝")

# --- 1. FONCTIONS UTILITAIRES (GÉNÉRATEUR) ---

def generer_svg_omr(titre, sous_titre, nb_questions):
    """Génère un SVG avec 4 marqueurs d'angle (Standard OMR)"""
    width = "210mm"
    height = "297mm"
    dwg = svgwrite.Drawing(size=(width, height))
    
    # Fond blanc
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
    
    # Titres
    dwg.add(dwg.text(titre, insert=("20mm", "20mm"), font_size="20px", font_family="Arial", font_weight="bold"))
    dwg.add(dwg.text(sous_titre, insert=("20mm", "30mm"), font_size="14px", font_family="Arial"))

    # --- STANDARD OMR : LES 4 CARRÉS D'ALIGNEMENT ---
    # Ces carrés aident l'algo à redresser la feuille
    # Position: 10mm des bords, Taille: 5mm
    markers = [
        ("10mm", "10mm"),   # Haut-Gauche
        ("195mm", "10mm"),  # Haut-Droit
        ("10mm", "282mm"),  # Bas-Gauche
        ("195mm", "282mm")  # Bas-Droit
    ]
    for mx, my in markers:
        dwg.add(dwg.rect(insert=(mx, my), size=("5mm", "5mm"), fill="black"))

    # Grille de questions (Centrage des lettres conservé)
    x_start_col1 = 30
    x_start_col2 = 115
    y_start = 50
    y_gap = 10
    
    for i in range(1, nb_questions + 1):
        # Gestion des colonnes (max 20 par colonne pour cet exemple)
        if i <= 20:
            x_pos = x_start_col1
            y_pos = y_start + ((i-1) * y_gap)
        else:
            x_pos = x_start_col2
            y_pos = y_start + ((i - 21) * y_gap)
            
        # Numéro de question
        dwg.add(dwg.text(f"{i}.", insert=(f"{x_pos}mm", f"{y_pos+5}mm"), font_size="12px", font_family="Arial"))
        
        # Bulles A B C D E
        options = ['A', 'B', 'C', 'D', 'E']
        for idx, opt in enumerate(options):
            cx = x_pos + 12 + (idx * 9) # Espacement horizontal
            cy = y_pos + 4              # Espacement vertical
            
            # Cercle
            dwg.add(dwg.circle(center=(f"{cx}mm", f"{cy}mm"), r="3mm", stroke="black", fill="white", stroke_width=1))
            # Lettre (Ajustement fin pour centrage)
            dwg.add(dwg.text(opt, insert=(f"{cx-1}mm", f"{cy+1.2}mm"), font_size="8px", font_family="Arial", fill="gray"))

    return dwg.tostring()

# --- 2. FONCTIONS DE VISION (CORRECTEUR) ---

def order_points(pts):
    """Ordonne les 4 points du contour trouvé"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

def four_point_transform(image, pts):
    """Redresse l'image selon les 4 points"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calcul largeur/hauteur max
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def trouver_feuille(image_cv):
    """
    Stratégie V3 : Chercher le plus grand contour rectangulaire.
    Cela marche si le scan a un fond contrasté (table sombre) OU 
    si les 4 coins noirs forment un contour implicite fort.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny détecte les bords
    edged = cv2.Canny(blurred, 75, 200)

    # Trouver les contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    docCnt = None
    
    # On cherche un polygone à 4 cotés
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Si le contour a 4 points et est assez grand
        if len(approx) == 4 and cv2.contourArea(c) > 1000:
            docCnt = approx
            break
            
    return docCnt, edged

def analyser_grille(warped_img):
    """Analyse basique de la noirceur"""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # Binarisation
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Simulation de note (Vraie logique à venir)
    pixels_noirs = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    ratio = pixels_noirs / total_pixels
    
    if ratio < 0.01:
        return thresh, "0/20 (Vide)", "Vierge"
    else:
        # Ici on simule une note basée sur la "quantité d'encre" pour l'instant
        return thresh, "Analyse...", "Copie remplie"

# --- 3. INTERFACE ---

st.sidebar.title("EdTech OMR V3")
mode = st.sidebar.radio("Navigation", ["Générateur de Gabarit", "Correcteur OMR"])

if mode == "Générateur de Gabarit":
    st.title("🖨️ Générateur (Mode 4 Coins)")
    
    with st.form("gen_form"):
        col1, col2 = st.columns(2)
        with col1:
            titre = st.text_input("Titre", "Examen Sciences")
        with col2:
            # RETOUR DU SLIDER DEMANDÉ
            nb_questions = st.number_input("Nombre de questions", min_value=5, max_value=40, value=20)
            
        sous_titre = st.text_input("Sous-titre", "Noircir les cases au crayon HB")
        submitted = st.form_submit_button("Générer PDF")
        
    if submitted:
        svg = generer_svg_omr(titre, sous_titre, nb_questions)
        pdf = cairosvg.svg2pdf(bytestring=svg.encode('utf-8'))
        
        st.success(f"Gabarit généré avec {nb_questions} questions.")
        
        # Aperçu
        preview = convert_from_bytes(pdf)[0]
        st.image(preview, caption="Aperçu (4 ancres)", width=350)
        
        # Téléchargement
        st.download_button("Télécharger PDF", pdf, "gabarit_omr.pdf", "application/pdf")

elif mode == "Correcteur OMR":
    st.title("📝 Correcteur")
    st.info("Astuce : Pour une meilleure détection, le scan doit montrer les 4 coins noirs de la feuille.")
    
    fichiers = st.file_uploader("Scans", accept_multiple_files=True, type=['pdf', 'jpg', 'png'])
    
    if fichiers:
        resultats = []
        for f in fichiers:
            # Conversion
            bytes_data = f.read()
            if f.type == "application/pdf":
                pil_img = convert_from_bytes(bytes_data)[0].convert('RGB')
                img = np.array(pil_img)[:, :, ::-1].copy()
            else:
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)

            # Détection
            docCnt, debug_edge = trouver_feuille(img)
            
            if docCnt is not None:
                # On dessine ce qu'on a trouvé (Vert = Succès)
                debug_vis = img.copy()
                cv2.drawContours(debug_vis, [docCnt], -1, (0, 255, 0), 5)
                
                warped = four_point_transform(img, docCnt.reshape(4, 2))
                img_analysee, note, status = analyser_grille(warped)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(debug_vis, cv2.COLOR_BGR2RGB), caption="Feuille détectée", width=300)
                with col2:
                    st.image(img_analysee, caption="Vision Machine", width=300)
                
                resultats.append({"Fichier": f.name, "Note": note, "Status": status})
            else:
                st.warning(f"Échec détection sur {f.name}")
                st.image(debug_edge, caption="Contours vus par l'IA", width=300)
                resultats.append({"Fichier": f.name, "Note": "Erreur", "Status": "Non détecté"})

        if resultats:
            st.dataframe(pd.DataFrame(resultats))
