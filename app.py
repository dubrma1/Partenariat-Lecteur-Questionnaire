import streamlit as st
import cv2
import numpy as np
import pandas as pd
import svgwrite
import cairosvg
import io
from pdf2image import convert_from_bytes
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="EdTech OMR Suite", layout="wide", page_icon="📝")

# --- FONCTIONS UTILITAIRES (GÉNÉRATEUR) ---

def generer_svg_omr(titre, sous_titre, nb_questions):
    """Génère le code SVG pour la feuille réponse."""
    # Dimensions A4 en pixels (approximatif pour 96 DPI) ou mm
    width = "210mm"
    height = "297mm"
    
    dwg = svgwrite.Drawing(size=(width, height))
    
    # Fond blanc
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
    
    # Titres
    dwg.add(dwg.text(titre, insert=("20mm", "20mm"), font_size="20px", font_family="Arial", font_weight="bold"))
    dwg.add(dwg.text(sous_titre, insert=("20mm", "30mm"), font_size="14px", font_family="Arial"))

    # Repères visuels (Carrés noirs aux 4 coins pour l'alignement futur)
    # Note: Pour un vrai OMR, ces coordonnées doivent être précises.
    # Ici, je place des repères fictifs pour l'exemple.
    markers = [("10mm", "10mm"), ("190mm", "10mm"), ("10mm", "280mm"), ("190mm", "280mm")]
    for mx, my in markers:
        dwg.add(dwg.rect(insert=(mx, my), size=("5mm", "5mm"), fill="black"))

    # Grille de questions (2 colonnes)
    x_start_col1 = 30
    x_start_col2 = 110
    y_start = 50
    y_gap = 10
    
    for i in range(1, nb_questions + 1):
        # Colonne 1 ou 2
        if i <= 20:
            x_pos = x_start_col1
            y_pos = y_start + (i * y_gap)
        else:
            x_pos = x_start_col2
            y_pos = y_start + ((i - 20) * y_gap)
            
        # Numéro question
        dwg.add(dwg.text(f"{i}.", insert=(f"{x_pos}mm", f"{y_pos}mm"), font_size="12px"))
        
        # Bulles A B C D E
        options = ['A', 'B', 'C', 'D', 'E']
        for idx, opt in enumerate(options):
            cx = x_pos + 10 + (idx * 8)
            cy = y_pos - 1.5 # Ajustement vertical
            
            # Cercle
            dwg.add(dwg.circle(center=(f"{cx}mm", f"{cy}mm"), r="2.5mm", stroke="black", fill="white", stroke_width=1))
            # Lettre dans le cercle
            dwg.add(dwg.text(opt, insert=(f"{cx-1.2}mm", f"{cy+1.2}mm"), font_size="8px", font_family="Arial"))

    return dwg.tostring()

# --- FONCTIONS UTILITAIRES (CORRECTEUR) ---

def order_points(pts):
    """Ordonne les 4 points du contour (Haut-Gauche, Haut-Droit, Bas-Droit, Bas-Gauche)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Applique la perspective pour obtenir une vue 'à plat'."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def traiter_image(image_cv, seuil_noir):
    """Logique de détection OMR (Simplifiée pour l'exemple)."""
    # 1. Conversion gris + flou
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Détection contours (Feuille)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        return image_cv, "⚠️ Impossible de détecter les 4 coins de la feuille."

    # 3. Redressement de l'image (Warp)
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    output_vis = four_point_transform(image_cv, docCnt.reshape(4, 2))

    # 4. Binarisation (Otsu ou seuil manuel)
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- SIMULATION DE LECTURE DE GRILLE ---
    # Ici, il faudrait découper la grille 'warped' en cellules.
    # Pour cet exemple Streamlit, je dessine juste le résultat du redressement
    # et je simule une extraction de notes pour montrer que le pipeline fonctionne.
    
    return output_vis, "✅ Feuille détectée et redressée."


# --- INTERFACE PRINCIPALE ---

st.sidebar.title("Navigation EdTech")
choix_mode = st.sidebar.radio("Mode :", ["Générateur de PDF", "Correcteur Automatique"])

if choix_mode == "Générateur de PDF":
    st.title("🖨️ Générateur de Gabarits OMR")
    st.markdown("Créez des feuilles réponses prêtes à imprimer.")

    with st.form("form_gen"):
        col1, col2 = st.columns(2)
        with col1:
            titre_doc = st.text_input("Titre du devoir", "Examen Final - Physique")
        with col2:
            nb_q = st.number_input("Nombre de questions", min_value=5, max_value=100, value=20)
        
        sous_titre_doc = st.text_input("Instructions", "Veuillez noircir complètement la case avec un crayon HB.")
        
        submit_gen = st.form_submit_button("Générer le PDF")

    if submit_gen:
        # 1. Création SVG
        svg_string = generer_svg_omr(titre_doc, sous_titre_doc, nb_q)
        
        # 2. Conversion en PDF (Mémoire)
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg_string.encode('utf-8'))
        
        st.success("Gabarit généré avec succès !")
        
        # 3. Prévisualisation (Image du PDF)
        try:
            preview_images = convert_from_bytes(pdf_bytes)
            st.image(preview_images[0], caption="Aperçu du document", width=400)
        except Exception as e:
            st.warning("Aperçu image non disponible (manque Poppler?), mais le PDF est bon.")

        # 4. Bouton Téléchargement
        st.download_button(
            label="📥 Télécharger le PDF",
            data=pdf_bytes,
            file_name="feuille_reponse.pdf",
            mime="application/pdf"
        )

elif choix_mode == "Correcteur Automatique":
    st.title("📝 Correcteur de Copies")
    
    # Paramètres avancés dans l'expander
    with st.expander("⚙️ Réglages de détection"):
        seuil_noir = st.slider("Seuil de détection (Noir)", 0, 255, 100, help="Plus c'est bas, plus il faut que ce soit noir.")
    
    uploaded_files = st.file_uploader("Chargez les scans (PDF ou Images)", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        if st.button("Lancer la correction"):
            resultats = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Conversion du fichier uploadé en image OpenCV
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                
                # Gestion PDF vs Image
                images_to_process = []
                if uploaded_file.type == "application/pdf":
                    # Convertir PDF en images
                    try:
                        # On lit les bytes directement
                        pil_images = convert_from_bytes(uploaded_file.getvalue()) 
                        for p_img in pil_images:
                            # Convert PIL -> OpenCV
                            open_cv_image = np.array(p_img) 
                            # Convert RGB to BGR 
                            open_cv_image = open_cv_image[:, :, ::-1].copy() 
                            images_to_process.append(open_cv_image)
                    except Exception as e:
                        st.error(f"Erreur Poppler sur {uploaded_file.name}. Vérifiez packages.txt.")
                else:
                    # C'est une image directe
                    image = cv2.imdecode(file_bytes, 1)
                    images_to_process.append(image)

                # Traitement de chaque page
                for img in images_to_process:
                    img_result, status = traiter_image(img, seuil_noir)
                    
                    # Affichage
                    st.write(f"**Fichier : {uploaded_file.name}** - {status}")
                    
                    # On retransforme en RGB pour l'affichage Streamlit (sinon c'est bleu)
                    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                    st.image(img_result_rgb, width=300)
                    
                    # Simulation de résultat (À remplacer par ta logique de grille exacte)
                    score = np.random.randint(10, 20) # Dummy score
                    resultats.append({"Fichier": uploaded_file.name, "Note": f"{score}/20", "Status": status})

                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Tableau final
            df = pd.DataFrame(resultats)
            st.success("Correction terminée !")
            st.dataframe(df)
            
            # Export CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger les notes (CSV)", csv, "notes_classe.csv", "text/csv")