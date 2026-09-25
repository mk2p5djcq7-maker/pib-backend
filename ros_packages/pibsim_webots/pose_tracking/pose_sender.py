from pathlib import Path
import cv2
import json
import math
import socket
import time
import urllib.request

import mediapipe as mp
import numpy as np


# ============================================
# EINSTELLUNGEN
# ============================================

CAMERA_INDEX = 0

WEBOTS_IP = "127.0.0.1"
WEBOTS_PORT = 5005

# Wie sicher ein Körperpunkt erkannt sein muss
MIN_VISIBILITY = 0.55

# 0.1 = sehr weich/langsam
# 0.5 = schneller, aber unruhiger
SMOOTHING_ALPHA = 0.25


# ============================================
# MEDIAPIPE-MODELL
# ============================================

MODEL_PATH = Path(__file__).with_name(
    "pose_landmarker_full.task"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)


# ============================================
# MEDIAPIPE LANDMARK-INDIZES
# ============================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

RIGHT_ELBOW = 14
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24


def ensure_model():
    """
    Lädt das MediaPipe-Modell beim ersten Start herunter.
    """

    if MODEL_PATH.exists():
        return

    print("MediaPipe-Modell wird einmalig heruntergeladen ...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Modell gespeichert:")
    print(MODEL_PATH)


def vec(landmark):
    """
    Macht aus einem MediaPipe-Landmark
    einen NumPy-3D-Vektor.
    """

    return np.array(
        [
            landmark.x,
            landmark.y,
            landmark.z
        ],
        dtype=np.float64
    )


def unit(v):
    """
    Normalisiert einen Vektor auf Länge 1.
    """

    length = np.linalg.norm(v)

    if length < 1e-8:
        return None

    return v / length


def angle_between(a, b):
    """
    Berechnet den Winkel zwischen zwei Vektoren.
    Ergebnis in Grad.
    """

    a = unit(a)
    b = unit(b)

    if a is None or b is None:
        return None

    cos_angle = np.clip(
        np.dot(a, b),
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cos_angle)
    )


def calculate_right_arm(world):
    """
    Berechnet:

    - Schulter vor/zurück
    - Schulter seitlich
    - Ellbogenbeugung

    aus den 3D-Landmarks.
    """

    left_shoulder = vec(
        world[LEFT_SHOULDER]
    )

    right_shoulder = vec(
        world[RIGHT_SHOULDER]
    )

    right_elbow = vec(
        world[RIGHT_ELBOW]
    )

    right_wrist = vec(
        world[RIGHT_WRIST]
    )

    left_hip = vec(
        world[LEFT_HIP]
    )

    right_hip = vec(
        world[RIGHT_HIP]
    )


    # Mittelpunkt der Schultern
    middle_shoulders = (
        left_shoulder +
        right_shoulder
    ) / 2.0


    # Mittelpunkt der Hüften
    middle_hips = (
        left_hip +
        right_hip
    ) / 2.0


    # ----------------------------------------
    # Körper-Koordinatensystem
    # ----------------------------------------

    body_right = unit(
        right_shoulder -
        left_shoulder
    )

    body_down = unit(
        middle_hips -
        middle_shoulders
    )

    if body_right is None or body_down is None:
        return None


    body_forward = unit(
        np.cross(
            body_right,
            body_down
        )
    )


    # Oberarm
    upper_arm = unit(
        right_elbow -
        right_shoulder
    )

    if body_forward is None or upper_arm is None:
        return None


    # ----------------------------------------
    # Komponenten des Oberarms
    # ----------------------------------------

    down_component = float(
        np.dot(
            upper_arm,
            body_down
        )
    )

    side_component = float(
        np.dot(
            upper_arm,
            body_right
        )
    )

    front_component = float(
        np.dot(
            upper_arm,
            body_forward
        )
    )


    # ----------------------------------------
    # Schulterwinkel
    # ----------------------------------------

    shoulder_front_deg = math.degrees(
        math.atan2(
            front_component,
            down_component
        )
    )

    shoulder_side_deg = math.degrees(
        math.atan2(
            side_component,
            down_component
        )
    )


    # ----------------------------------------
    # Ellbogenwinkel
    # ----------------------------------------

    elbow_inner_deg = angle_between(

        right_shoulder -
        right_elbow,

        right_wrist -
        right_elbow
    )

    if elbow_inner_deg is None:
        return None


    # Gerade = 0°
    # Gebeugt = z.B. 90°
    elbow_bend_deg = (
        180.0 -
        elbow_inner_deg
    )


    # ----------------------------------------
    # Begrenzungen
    # ----------------------------------------

    return {

        "shoulder_front_deg":
            float(
                np.clip(
                    shoulder_front_deg,
                    -90,
                    150
                )
            ),

        "shoulder_side_deg":
            float(
                np.clip(
                    shoulder_side_deg,
                    -30,
                    150
                )
            ),

        "elbow_bend_deg":
            float(
                np.clip(
                    elbow_bend_deg,
                    0,
                    150
                )
            )
    }


def smooth_angles(previous, current):
    """
    Glättet die Kamerawerte.
    """

    if previous is None:
        return current.copy()

    result = {}

    for key in current:

        result[key] = (

            (1.0 - SMOOTHING_ALPHA)
            * previous[key]

            +

            SMOOTHING_ALPHA
            * current[key]
        )

    return result


def visible_enough(landmarks):
    """
    Prüft, ob alle benötigten Körperpunkte
    zuverlässig sichtbar sind.
    """

    required = [

        LEFT_SHOULDER,
        RIGHT_SHOULDER,

        RIGHT_ELBOW,
        RIGHT_WRIST,

        LEFT_HIP,
        RIGHT_HIP
    ]

    return all(

        getattr(
            landmarks[i],
            "visibility",
            1.0
        ) >= MIN_VISIBILITY

        for i in required
    )


def draw_keypoints(frame, landmarks):
    """
    Zeichnet unseren Arm und Oberkörper
    ins Kamerabild.
    """

    height, width = frame.shape[:2]


    def point(index):

        landmark = landmarks[index]

        return (

            int(
                landmark.x *
                width
            ),

            int(
                landmark.y *
                height
            )
        )


    connections = [

        (LEFT_SHOULDER, RIGHT_SHOULDER),

        (RIGHT_SHOULDER, RIGHT_ELBOW),

        (RIGHT_ELBOW, RIGHT_WRIST),

        (LEFT_SHOULDER, LEFT_HIP),

        (RIGHT_SHOULDER, RIGHT_HIP),

        (LEFT_HIP, RIGHT_HIP)
    ]


    for start, end in connections:

        cv2.line(
            frame,
            point(start),
            point(end),
            (0, 255, 0),
            2
        )


    important_points = [

        LEFT_SHOULDER,
        RIGHT_SHOULDER,

        RIGHT_ELBOW,
        RIGHT_WRIST,

        LEFT_HIP,
        RIGHT_HIP
    ]


    for index in important_points:

        cv2.circle(
            frame,
            point(index),
            5,
            (0, 0, 255),
            -1
        )


def main():

    ensure_model()


    # ========================================
    # UDP
    # ========================================

    udp = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    target = (
        WEBOTS_IP,
        WEBOTS_PORT
    )


    # ========================================
    # KAMERA
    # ========================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Kamera konnte nicht geöffnet werden. "
            "CAMERA_INDEX prüfen oder andere "
            "Kamera-App schließen."
        )


    # ========================================
    # MEDIAPIPE
    # ========================================

    BaseOptions = (
        mp.tasks.BaseOptions
    )

    PoseLandmarker = (
        mp.tasks.vision.PoseLandmarker
    )

    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )


    options = PoseLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),

        running_mode=
            RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )


    filtered = None


    with PoseLandmarker.create_from_options(
        options
    ) as landmarker:


        while True:


            # =================================
            # Kamerabild holen
            # =================================

            success, frame = cap.read()


            if not success:

                print(
                    "Kein Kamerabild erhalten."
                )

                break


            # OpenCV = BGR
            # MediaPipe = RGB

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(

                image_format=
                    mp.ImageFormat.SRGB,

                data=rgb
            )


            timestamp_ms = int(
                time.perf_counter()
                * 1000
            )


            # =================================
            # Pose erkennen
            # =================================

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            status = (
                "Keine Pose erkannt"
            )


            if (
                result.pose_landmarks
                and
                result.pose_world_landmarks
            ):


                image_landmarks = (
                    result.pose_landmarks[0]
                )


                world_landmarks = (
                    result.pose_world_landmarks[0]
                )


                draw_keypoints(
                    frame,
                    image_landmarks
                )


                # =============================
                # Sind Arm und Körper sichtbar?
                # =============================

                if visible_enough(
                    image_landmarks
                ):


                    angles = (
                        calculate_right_arm(
                            world_landmarks
                        )
                    )


                    if angles is not None:


                        # =====================
                        # Werte glätten
                        # =====================

                        filtered = (
                            smooth_angles(
                                filtered,
                                angles
                            )
                        )


                        # =====================
                        # Daten senden
                        # =====================

                        message = json.dumps(
                            filtered
                        ).encode(
                            "utf-8"
                        )


                        udp.sendto(
                            message,
                            target
                        )


                        status = (

                            f"vorne "
                            f"{filtered['shoulder_front_deg']:+.0f} deg | "

                            f"seitlich "
                            f"{filtered['shoulder_side_deg']:+.0f} deg | "

                            f"Ellbogen "
                            f"{filtered['elbow_bend_deg']:.0f} deg"
                        )

                else:

                    status = (
                        "Arm nicht sicher sichtbar"
                    )


            # =================================
            # Bild anzeigen
            # =================================

            display = cv2.flip(
                frame,
                1
            )


            cv2.putText(

                display,

                status,

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


            cv2.putText(

                display,

                "Q oder ESC zum Beenden",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2
            )


            cv2.imshow(
                "pib Pose Tracking",
                display
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key in (
                ord("q"),
                27
            ):
                break


    cap.release()

    cv2.destroyAllWindows()

    udp.close()


if __name__ == "__main__":
    main()