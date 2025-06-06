import cv2
import numpy as np
import dlib
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove
import os
import urllib.request
import uuid
from scipy.spatial import Delaunay
import subprocess


selected_image = None

class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.load_models()

        self.source_image = None
        self.target_image = None
        self.result_image = None
        self.source_path = ""
        self.target_path = ""

        self.display_width = 400
        self.display_height = 300

    def setup_ui(self):
        self.root.title("Professional Face Swap v2.3")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)

        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.image_frame = Frame(main_frame)
        self.image_frame.pack(fill=BOTH, expand=True)

        self.source_frame = LabelFrame(self.image_frame, text="Source Image")
        self.source_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.source_label = Label(self.source_frame)
        self.source_label.pack()

        self.target_frame = LabelFrame(self.image_frame, text="Target Image")
        self.target_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.target_label = Label(self.target_frame)
        self.target_label.pack()

        self.result_frame = LabelFrame(self.image_frame, text="Result Image")
        self.result_label = Label(self.result_frame)
        self.result_label.pack()

        control_frame = Frame(main_frame)
        control_frame.pack(fill=X, pady=10)

        Button(control_frame, text="Load Source", command=self.load_source).grid(row=0, column=0, padx=5)
        Button(control_frame, text="Load Target", command=self.load_target).grid(row=0, column=1, padx=5)
        Button(control_frame, text="Swap Faces", command=self.swap_faces).grid(row=0, column=2, padx=5)

        self.save_button = Button(control_frame, text="Save Result", command=self.save_result, state=DISABLED)
        self.save_button.grid(row=0, column=3, padx=5)

        Button(control_frame, text="Generate AI Face", command=self.generate_ai_face).grid(row=0, column=4, padx=5)
        Button(control_frame, text="Webcam Source", command=lambda: self.capture_from_webcam(is_source=True)).grid(
            row=0, column=5, padx=5)
        Button(control_frame, text="Webcam Target", command=lambda: self.capture_from_webcam(is_source=False)).grid(
            row=0, column=6, padx=5)
        Button(control_frame, text="Live Swap Video", command=self.open_live_video).grid(row=0, column=7, padx=5)
        """
        Button(control_frame, text="Swap avec SimSwap", command=self.run_simswap_gui).grid(row=2, column=0, padx=5)
        """
        Button(control_frame, text="Deepfake avec DeepFaceLive", command=self.run_deepfacelive).grid(row=2, column=1, padx=5)

        settings_frame = Frame(control_frame)
        settings_frame.grid(row=1, column=0, columnspan=8, pady=5)

        Label(settings_frame, text="Blend Amount:").grid(row=0, column=0)
        self.blend_scale = Scale(settings_frame, from_=0, to=100, orient=HORIZONTAL, length=150)
        self.blend_scale.set(65)
        self.blend_scale.grid(row=0, column=1)
        self.blend_scale.bind("<ButtonRelease-1>", self.update_blend)

        Label(settings_frame, text="Color Adjustment:").grid(row=0, column=2)
        self.color_scale = Scale(settings_frame, from_=0, to=100, orient=HORIZONTAL, length=150)
        self.color_scale.set(50)
        self.color_scale.grid(row=0, column=3)
        self.color_scale.bind("<ButtonRelease-1>", self.update_color)

        self.status_var = StringVar()
        self.status_var.set("Ready to load images")
        status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief=SUNKEN, anchor=W)
        status_bar.pack(side=BOTTOM, fill=X)

        for i in range(3):
            self.image_frame.columnconfigure(i, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

    def load_models(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            model_path = "C:/Users/pa76wlu/Desktop/PythonProject1/models/shape_predictor_68_face_landmarks.dat" # A adapter
            if not os.path.exists(model_path):
                raise FileNotFoundError("Dlib model file not found.")
            self.predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            self.root.destroy()

    def load_image(self, is_source=True):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Invalid image file")
            else :
                selected_image = image
                if is_source:
                    self.source_image = image
                    self.source_path = path
                    self.show_image(image, self.source_label)
                    self.status_var.set(f"Source image loaded: {os.path.basename(path)}")
                else:
                    self.target_image = image
                    self.target_path = path
                    self.show_image(image, self.target_label)
                    self.status_var.set(f"Target image loaded: {os.path.basename(path)}")
                if self.source_image is not None and self.target_image is not None:
                    self.status_var.set("Ready to perform face swap")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_source(self):
        self.load_image(is_source=True)

    def load_target(self):
        self.load_image(is_source=False)

    def capture_from_webcam(self, is_source=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        self.status_var.set("SPACE: Capture image | ESC: Exit")
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        captured = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1)
            if key == 32:
                captured = frame.copy()
                break
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured is not None:
            if is_source:
                self.source_image = captured
                self.source_path = "webcam_source.jpg"
                self.show_image(captured, self.source_label)
                self.status_var.set("Source image captured from webcam.")
            else:
                self.target_image = captured
                self.target_path = "webcam_target.jpg"
                self.show_image(captured, self.target_label)
                self.status_var.set("Target image captured from webcam.")
            if self.source_image is not None and self.target_image is not None:
                self.status_var.set("Ready to perform face swap.")

    def generate_ai_face(self):
        try:
            self.status_var.set("Downloading AI face...")
            self.root.config(cursor="watch")
            self.root.update()

            url = "https://thispersondoesnotexist.com/"
            folder = os.path.join(os.getcwd(), "ai_faces")
            os.makedirs(folder, exist_ok=True)

            filename = f"ai_face_{uuid.uuid4().hex[:8]}.jpg"
            filepath = os.path.join(folder, filename)

            urllib.request.urlretrieve(url, filepath)
            image = cv2.imread(filepath)

            if image is None:
                raise ValueError("AI face could not be downloaded.")

            self.source_image = image
            self.source_path = filepath
            self.show_image(image, self.source_label)
            self.status_var.set("AI face loaded.")
            selected_image = image
            if self.target_image is not None:
                self.status_var.set("Ready to perform face swap with AI face.")
        except Exception as e:
            messagebox.showerror("AI Face Error", str(e))
            self.status_var.set("AI face generation failed.")
        finally:
            self.root.config(cursor="")

    def remove_background(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = remove(img_rgb)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        shape = self.predictor(gray, faces[0])
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)

    def create_mask(self, landmarks, shape):
        hull = cv2.convexHull(landmarks)
        mask = np.zeros(shape[:2], dtype=np.float32)
        cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask[..., np.newaxis]

    def run_deepfacelive(self):
        if self.source_image is None:
            messagebox.showerror("Erreur", "Veuillez charger une image source.")
            return

        try:
            self.status_var.set("Préparation de DeepFaceLive...")
            self.root.config(cursor="watch")
            self.root.update()

            # Retire le fond
            img_no_bg = remove(self.source_image)

            # Sauvegarde l'image temporairement dans le bon format
            face_dir = os.path.join(os.getcwd(), "deepfacelivetemp", "assets", "faces")
            os.makedirs(face_dir, exist_ok=True)
            face_path = os.path.join(face_dir, "input_face.png")
            cv2.imwrite(face_path, img_no_bg)

            # Lance DeepFaceLive en mode live webcam
            deepfacelive_path = "C:/Users/pa76wlu/Desktop/PythonProject1/DeepFaceLive-master/"  # adapte ce chemin
            #subprocess.Popen(["python", "main.py"], cwd=deepfacelive_path)
            script_path = os.path.join(deepfacelive_path, "main.py")
            subprocess.Popen([
                "python", "main.py", "run", "DeepFaceLive",
                "--userdata-dir", "workspace"
            ], cwd=deepfacelive_path)

            """subprocess.Popen([
                "python", script_path,
                #"python", "main.py"
                "--face", face_path,
                "--source", "webcam"
            ], cwd=deepfacelive_path)"""
            self.status_var.set("DeepFaceLive en cours d'exécution.")
        except Exception as e:
            messagebox.showerror("Erreur DeepFaceLive", str(e))
            self.status_var.set("Échec lancement DeepFaceLive.")
        finally:
            self.root.config(cursor="")

    def run_facefusion_gui(self):
        if self.source_path and self.target_path:
            output_path = os.path.join(os.getcwd(), "facefusion_output.jpg")
            facefusion_dir = "C:/Users/pa76wlu/Desktop/PythonProject1/FaceFusion"  # à adapter

            command = [
                "python",
                os.path.join(facefusion_dir, "run.py"),
                "--source", self.source_path,
                "--target", self.target_path,
                "--output", output_path
            ]

            try:
                subprocess.run(command, check=True)
                if os.path.exists(output_path):
                    result = cv2.imread(output_path)
                    if result is not None:
                        self.result_image = result
                        self.show_result()
                        self.save_button.config(state=NORMAL)
                        self.status_var.set("Face swap avec FaceFusion terminé.")
                    else:
                        raise ValueError("Résultat illisible")
                else:
                    raise FileNotFoundError("L'image de sortie n'a pas été générée.")
            except Exception as e:
                messagebox.showerror("FaceFusion Error", str(e))
                self.status_var.set("FaceFusion échoué.")

    def adjust_colors(self, src, target, amount):
        if amount == 0:
            return src
        try:
            src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

            src_mean, src_std = cv2.meanStdDev(src_lab)
            tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

            src_mean, src_std = src_mean.flatten(), src_std.flatten()
            tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

            src_std[src_std == 0] = 1.0
            normalized = (src_lab - src_mean) / src_std
            adjusted = normalized * ((1 - amount) * src_std + amount * tgt_std) + \
                       ((1 - amount) * src_mean + amount * tgt_mean)
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
        except Exception as e:
            messagebox.showerror("Error", f"Color adjustment failed: {str(e)}")
            return src

    def warp_image_3d(self, src_img, src_points, tgt_points, img_shape):
        warped_img = np.zeros(img_shape, dtype=src_img.dtype)
        tri = Delaunay(tgt_points)

        for tri_indices in tri.simplices:
            t1 = np.float32([src_points[i] for i in tri_indices])
            t2 = np.float32([tgt_points[i] for i in tri_indices])
            r1 = cv2.boundingRect(t1)
            r2 = cv2.boundingRect(t2)
            t1_rect = [(p[0] - r1[0], p[1] - r1[1]) for p in t1]
            t2_rect = [(p[0] - r2[0], p[1] - r2[1]) for p in t2]
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16)
            img1_rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            size = (r2[2], r2[3])
            warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
            warped_triangle = cv2.warpAffine(img1_rect, warp_mat, size, None, flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT_101)
            warped_patch = warped_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
            warped_patch = warped_patch * (1 - mask) + warped_triangle * mask
            warped_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = warped_patch
        return warped_img

    def swap_faces(self):
        if self.source_image is None or self.target_image is None:
            messagebox.showerror("Error", "Please load both source and target images.")
            return
        self.status_var.set("Processing... Please wait.")
        self.root.config(cursor="watch")
        self.root.update()
        try:
            src_points = self.get_landmarks(self.source_image)
            tgt_points = self.get_landmarks(self.target_image)
            if src_points is None or tgt_points is None:
                raise ValueError("Face not detected in one or both images.")
            warped = self.warp_image_3d(self.source_image, src_points, tgt_points, self.target_image.shape)
            mask = self.create_mask(tgt_points, self.target_image.shape)
            center = tuple(np.mean(tgt_points, axis=0).astype(int))
            self.result_image = cv2.seamlessClone(
                warped,
                self.target_image,
                (mask * 255).astype(np.uint8),
                center,
                cv2.NORMAL_CLONE
            )
            self.show_result()
            self.save_button.config(state=NORMAL)
            self.status_var.set("Face swap completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Face swap failed: {str(e)}")
            self.status_var.set("Face swap failed.")
        finally:
            self.root.config(cursor="")

    """
    def run_simswap_gui(self):
        if self.source_path and self.target_path:
            output_folder = os.path.join(os.getcwd(), "simswap_output")
            os.makedirs(output_folder, exist_ok=True)

            command = [
                "python", "run_simswap.py",
                "--target", self.target_path,
                "--source", self.source_path,
                "--output", output_folder
            ]
            subprocess.run(command)
            result_img_path = os.path.join(output_folder, "result.jpg")
            if os.path.exists(result_img_path):
                result = cv2.imread(result_img_path)
                self.result_image = result
                self.show_result()
                self.save_button.config(state=NORMAL)
                self.status_var.set("Face swap avec SimSwap terminé.")
    """

    """
    def run_facefusion_gui(self):
    if self.source_path and self.target_path:
        output_path = os.path.join(os.getcwd(), "facefusion_output.jpg")

        command = [
            "python", "run_facefusion.py",
            "--source", self.source_path,
            "--target", self.target_path,
            "--output", output_path
        ]

        subprocess.run(command)
        if os.path.exists(output_path):
            result = cv2.imread(output_path)
            self.result_image = result
            self.show_result()
            self.save_button.config(state=NORMAL)
            self.status_var.set("Face swap avec FaceFusion terminé.")
    """

    def update_face_swap(self):
        if not hasattr(self, 'warped_src'):
            return

        try:
            blend_amount = self.blend_scale.get() / 100.0
            color_amount = self.color_scale.get() / 100.0

            if color_amount > 0:
                color_adjusted = self.adjust_colors(self.warped_src, self.target_image, color_amount)
            else:
                color_adjusted = self.warped_src

            mask_3ch = np.repeat(self.mask, 3, axis=2)
            blended = (color_adjusted * mask_3ch + self.target_image * (1 - mask_3ch)).astype(np.uint8)
            self.result_image = (blended * blend_amount + self.target_image * (1 - blend_amount)).astype(np.uint8)

            self.show_result()
            self.save_button.config(state=NORMAL)
            self.status_var.set("Face swap completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Update failed: {str(e)}")

    def update_blend(self, event=None):
        if hasattr(self, 'result_image'):
            self.update_face_swap()

    def update_color(self, event=None):
        if hasattr(self, 'result_image'):
            self.update_face_swap()

    def show_image(self, image, label_widget):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized))
        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    def show_result(self):
        self.result_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.image_frame.columnconfigure(2, weight=1)
        self.show_image(self.result_image, self.result_label)

    def save_result(self):
        if not hasattr(self, 'result_image'):
            messagebox.showerror("Error", "No result image to save.")
            return
        default_name = f"swap_{os.path.basename(self.source_path)}_{os.path.basename(self.target_path)}"
        path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            try:
                cv2.imwrite(path, self.result_image)
                messagebox.showinfo("Saved", f"Image saved at:\n{path}")
                self.status_var.set(f"Saved to {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

    def open_live_video(self):
        if self.source_image is None:
            messagebox.showwarning("Live Swap", "Please load a source image first.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam.")
            return

        self.status_var.set("Live video started. Press ESC to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.perform_live_swap(frame, self.source_image)
            cv2.imshow("Live Face Swap", result)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Live video closed.")

    def perform_live_swap(self, frame, source_image):
        src_landmarks = self.get_landmarks(source_image)
        tgt_landmarks = self.get_landmarks(frame)

        if src_landmarks is None or tgt_landmarks is None:
            return frame

        matrix, _ = cv2.estimateAffinePartial2D(src_landmarks, tgt_landmarks)
        warped_src = cv2.warpAffine(source_image, matrix, (frame.shape[1], frame.shape[0]))

        hull = cv2.convexHull(tgt_landmarks)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        mask3 = cv2.merge([mask, mask, mask]) / 255.0
        blended = (warped_src * mask3 + frame * (1 - mask3)).astype(np.uint8)

        return blended


if __name__ == "__main__":
    root = Tk()
    app = FaceSwapApp(root)
    root.mainloop()
