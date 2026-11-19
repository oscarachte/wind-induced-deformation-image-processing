import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import os

# ============================================================
# DOSSIER DU SCRIPT
# ============================================================


script_dir = os.path.dirname(os.path.abspath(__file__))

print("Dossier du script :", script_dir)
print("Dossier courant (terminal) :", os.getcwd())



# ============================================================
# PARAMETRES A MODIFIER
# ============================================================

video_path = os.path.join(script_dir, "video_tige.mp4")
mm_per_pixel = 0.109

# Afficher les images pendant le traitement
show_preview = True

# Sauvegarder une video avec le suivi
save_output_video = True
output_video_path = os.path.join(script_dir, "tracking_output.mp4")

# Zone d'interet (ROI) : x, y, largeur, hauteur
# Mettez None si vous voulez traiter l'image entiere
roi = None
# Exemple :
# roi = (1100, 80, 500, 420)

# Petit seuil de surface pour ignorer les parasites
min_area = 100

# ============================================================
# FONCTIONS UTILES
# ============================================================

def get_red_mask(hsv):
    """
    Cree un masque pour detecter le rouge en HSV.
    Le rouge est souvent sur 2 intervalles.
    """
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    # Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def find_largest_contour(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if not valid_contours:
        return None

    largest = max(valid_contours, key=cv2.contourArea)
    return largest

def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_tip_from_contour(contour, base_point):
    """
    Trouve le point du contour le plus eloigne de la base.
    """
    contour_points = contour[:, 0, :]  # shape (N, 2)

    max_dist = -1
    tip_point = None

    for pt in contour_points:
        d = euclidean_distance((pt[0], pt[1]), base_point)
        if d > max_dist:
            max_dist = d
            tip_point = (int(pt[0]), int(pt[1]))

    return tip_point

def save_summary_table_png(summary_rows, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.axis("off")

    table = ax.table(
        cellText=summary_rows,
        colLabels=["Information", "Valeur"],
        loc="center",
        cellLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# CHARGEMENT VIDEO
# ============================================================

print("Dossier du script :", script_dir)
print("Chemin video :", video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Impossible d'ouvrir la video : {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS : {fps}")
print(f"Nombre de frames : {frame_count}")
print(f"Resolution : {width} x {height}")

# ============================================================
# PREPARATION SORTIE VIDEO
# ============================================================

writer = None
if save_output_video:
    if roi is None:
        out_w, out_h = width, height
    else:
        out_w, out_h = roi[2], roi[3]

    # mp4v + .mp4 = souvent plus compatible avec QuickTime
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

# ============================================================
# ETAPE 1 : CHOISIR LA BASE SUR LA PREMIERE FRAME
# ============================================================

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Impossible de lire la premiere frame.")

if roi is not None:
    x_roi, y_roi, w_roi, h_roi = roi
    frame_for_base = first_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
else:
    x_roi, y_roi = 0, 0
    frame_for_base = first_frame.copy()

base_point_clicked = []

def select_base(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        base_point_clicked.clear()
        base_point_clicked.append((x, y))

cv2.namedWindow("Choisir la base")
cv2.setMouseCallback("Choisir la base", select_base)

print("Cliquez sur la base fixe de la tige, puis appuyez sur une touche.")

while True:
    display = frame_for_base.copy()
    if base_point_clicked:
        cv2.circle(display, base_point_clicked[0], 6, (0, 255, 0), -1)

    cv2.imshow("Choisir la base", display)
    key = cv2.waitKey(20) & 0xFF
    if key != 255 and base_point_clicked:
        break

cv2.destroyWindow("Choisir la base")

base_point = base_point_clicked[0]
print(f"Base selectionnee : {base_point}")

# Revenir au debut de la video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ============================================================
# ETAPE 2 : TRAITEMENT IMAGE PAR IMAGE
# ============================================================

results = []
initial_tip = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi is not None:
        frame_proc = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
    else:
        frame_proc = frame.copy()

    hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV)
    mask = get_red_mask(hsv)
    contour = find_largest_contour(mask, min_area=min_area)

    detected = False
    tip_point = None
    disp_px = np.nan
    disp_mm = np.nan

    if contour is not None:
        tip_point = find_tip_from_contour(contour, base_point)

        if tip_point is not None:
            detected = True

            if initial_tip is None:
                initial_tip = tip_point

            # deplacement de la pointe par rapport a la position initiale
            dx_px = tip_point[0] - initial_tip[0]
            dy_px = tip_point[1] - initial_tip[1]
            disp_px = sqrt(dx_px**2 + dy_px**2)
            disp_mm = disp_px * mm_per_pixel

            # Dessin
            cv2.drawContours(frame_proc, [contour], -1, (255, 0, 0), 2)
            cv2.circle(frame_proc, base_point, 6, (0, 255, 0), -1)
            cv2.circle(frame_proc, tip_point, 6, (0, 0, 255), -1)
            cv2.line(frame_proc, initial_tip, tip_point, (0, 255, 255), 2)

            cv2.putText(frame_proc, f"Frame: {frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_proc, f"Disp: {disp_mm:.3f} mm", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame_proc, "Pointe non detectee", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame_proc, "Tige non detectee", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    time_s = frame_idx / fps

    results.append({
        "frame": frame_idx,
        "time_s": time_s,
        "tip_x_px": tip_point[0] if tip_point is not None else np.nan,
        "tip_y_px": tip_point[1] if tip_point is not None else np.nan,
        "disp_px": disp_px,
        "disp_mm": disp_mm,
        "detected": detected
    })

    if show_preview:
        cv2.imshow("Mask", mask)
        cv2.imshow("Tracking", frame_proc)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Echap pour quitter
            break

    if writer is not None:
        writer.write(frame_proc)

    frame_idx += 1

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

# ============================================================
# OUVERTURE AUTOMATIQUE DE LA VIDEO SUR MAC
# ============================================================

if save_output_video and os.path.exists(output_video_path):
    print("Ouverture automatique de la video...")
    os.system(f"open '{output_video_path}'")

# ============================================================
# SAUVEGARDE RESULTATS
# ============================================================

csv_path = os.path.join(script_dir, "resultats_tracking.csv")
simple_csv_path = os.path.join(script_dir, "resultats_tracking_simple.csv")
resume_txt_path = os.path.join(script_dir, "resume_resultats.txt")
resume_png_path = os.path.join(script_dir, "resume_table.png")

df = pd.DataFrame(results)

# Fichier complet
df.to_csv(csv_path, index=False)

# Garder seulement les lignes detectees
valid_df = df.dropna(subset=["disp_mm"]).copy()

# Tableau simplifie
df_simple = valid_df[["frame", "time_s", "tip_x_px", "tip_y_px", "disp_px", "disp_mm"]].copy()
df_simple.columns = [
    "Frame",
    "Time (s)",
    "Tip X (px)",
    "Tip Y (px)",
    "Displacement (px)",
    "Displacement (mm)"
]

# Version lisible arrondie pour lecture
df_simple_rounded = df_simple.round({
    "Time (s)": 6,
    "Tip X (px)": 3,
    "Tip Y (px)": 3,
    "Displacement (px)": 6,
    "Displacement (mm)": 6
})

df_simple_rounded.to_csv(simple_csv_path, index=False)

print(f"\nResultats complets sauvegardes dans : {csv_path}")
print(f"Resultats simplifies sauvegardes dans : {simple_csv_path}")

# ============================================================
# RESUME LISIBLE ET PRECIS
# ============================================================

if len(df_simple) > 0:
    max_idx = df_simple["Displacement (mm)"].idxmax()
    max_row = df_simple.loc[max_idx]
    final_row = df_simple.iloc[-1]

    summary_text = f"""
================ RESUME DES RESULTATS ================

Nombre de frames detectees : {len(df_simple)}
Temps total analyse : {final_row['Time (s)']:.6f} s

Deplacement final :
  - en pixels : {final_row['Displacement (px)']:.6f} px
  - en mm     : {final_row['Displacement (mm)']:.6f} mm

Deplacement maximal :
  - frame     : {int(max_row['Frame'])}
  - temps     : {max_row['Time (s)']:.6f} s
  - en pixels : {max_row['Displacement (px)']:.6f} px
  - en mm     : {max_row['Displacement (mm)']:.6f} mm

Position de la pointe au maximum :
  - X = {max_row['Tip X (px)']:.3f} px
  - Y = {max_row['Tip Y (px)']:.3f} px

=====================================================
"""

    print(summary_text)

    print("Apercu du tableau :")
    print(df_simple_rounded.head(300).to_string(index=False))

    with open(resume_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\nApercu du tableau :\n")
        f.write(df_simple_rounded.head(300).to_string(index=False))

    print(f"\nResume sauvegarde dans : {resume_txt_path}")

    summary_rows = [
        ["Frames détectées", f"{len(df_simple)}"],
        ["Temps total analysé (s)", f"{final_row['Time (s)']:.6f}"],
        ["Déplacement final (px)", f"{final_row['Displacement (px)']:.6f}"],
        ["Déplacement final (mm)", f"{final_row['Displacement (mm)']:.6f}"],
        ["Frame du maximum", f"{int(max_row['Frame'])}"],
        ["Temps du maximum (s)", f"{max_row['Time (s)']:.6f}"],
        ["Déplacement max (px)", f"{max_row['Displacement (px)']:.6f}"],
        ["Déplacement max (mm)", f"{max_row['Displacement (mm)']:.6f}"],
        ["Tip X au max (px)", f"{max_row['Tip X (px)']:.3f}"],
        ["Tip Y au max (px)", f"{max_row['Tip Y (px)']:.3f}"],
    ]
    save_summary_table_png(summary_rows, resume_png_path)
    print(f"Tableau PNG sauvegarde dans : {resume_png_path}")

# ============================================================
# AFFICHAGE GRAPHIQUE AVEC POINT MAX
# ============================================================

if len(df_simple) > 0:
    max_idx = df_simple["Displacement (mm)"].idxmax()
    max_time = df_simple.loc[max_idx, "Time (s)"]
    max_disp = df_simple.loc[max_idx, "Displacement (mm)"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_simple["Time (s)"],
        df_simple["Displacement (mm)"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Déplacement"
    )

    plt.scatter(max_time, max_disp, color="red", s=120, zorder=6)

    text_box = (
        f"Max deformation\n"
        f"{max_disp:.3f} mm\n"
        f"à t = {max_time:.2f} s"
    )

    plt.text(
        0.98, 0.92,
        text_box,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.9)
    )

    plt.xlabel("Temps (s)")
    plt.ylabel("Déplacement de la pointe (mm)")
    plt.title("Déplacement de la pointe en fonction du temps", pad=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Aucune valeur de déplacement détectée, graphique non affiché.")

print("Chemin CSV complet :", csv_path)
print("Chemin CSV simple :", simple_csv_path)
print("Chemin resume txt :", resume_txt_path)
print("Chemin resume png :", resume_png_path)