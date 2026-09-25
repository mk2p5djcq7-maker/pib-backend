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

HAND_SMOOTHING_ALPHA = 0.12

FINGER_SMOOTHING_ALPHA = 0.12


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

HAND_MODEL_PATH = Path(__file__).with_name(
    "hand_landmarker.task"
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/"
    "hand_landmarker.task"
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

# ============================================
# HAND LANDMARKS
# ============================================

HAND_WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

HAND_CONNECTIONS = [

    # Handfläche
    (0, 1),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),

    # Daumen
    (1, 2),
    (2, 3),
    (3, 4),

    # Zeigefinger
    (5, 6),
    (6, 7),
    (7, 8),

    # Mittelfinger
    (9, 10),
    (10, 11),
    (11, 12),

    # Ringfinger
    (13, 14),
    (14, 15),
    (15, 16),

    # Kleiner Finger
    (17, 18),
    (18, 19),
    (19, 20)
]

def calculate_right_hand_angles(
    pose_world,
    hand_world
):
    """
    Berechnet zunächst die Rotation des
    rechten Unterarms aus Pose + Handtracking.

    Ergebnis:
    forearm_rotation_deg
    """

    # ========================================
    # POSE-PUNKTE
    # ========================================

    right_shoulder = vec(
        pose_world[RIGHT_SHOULDER]
    )

    right_elbow = vec(
        pose_world[RIGHT_ELBOW]
    )

    right_wrist = vec(
        pose_world[RIGHT_WRIST]
    )


    # ========================================
    # ARM-VEKTOREN
    # ========================================

    upper_arm = unit(
        right_elbow -
        right_shoulder
    )

    forearm_axis = unit(
        right_wrist -
        right_elbow
    )

    if (
        upper_arm is None
        or
        forearm_axis is None
    ):
        return None


    # ========================================
    # HAND-PUNKTE
    # ========================================

    hand_wrist = vec(
        hand_world[HAND_WRIST]
    )

    index_mcp = vec(
        hand_world[INDEX_MCP]
    )

    index_pip = vec(
    hand_world[INDEX_PIP]
    )

    index_dip = vec(
        hand_world[INDEX_DIP]
    )

    index_tip = vec(
        hand_world[INDEX_TIP]
    )

    middle_mcp = vec(
        hand_world[MIDDLE_MCP]
    )
    
    ring_mcp = vec(
    hand_world[RING_MCP]
    )

    pinky_mcp = vec(
        hand_world[PINKY_MCP]
    )


    # ========================================
    # HAND-KOORDINATENSYSTEM
    # ========================================

# ========================================
# STABILES HAND-KOORDINATENSYSTEM
# ========================================

# Mittelpunkt der vier Fingergrundgelenke.
# Das ist stabiler als nur der Mittelfinger.
    palm_front = (
        index_mcp
        + middle_mcp
        + ring_mcp
        + pinky_mcp
    ) / 4.0


    # ----------------------------------------
    # LÄNGSRICHTUNG DER HAND
    # ----------------------------------------

    hand_forward = unit(
        palm_front -
        hand_wrist
    )

    if hand_forward is None:
        return None


    # ----------------------------------------
    # QUERACHSE DER HAND
    # ----------------------------------------

    hand_side_raw = (
        index_mcp -
        pinky_mcp
    )


    # hand_side exakt rechtwinklig zu
    # hand_forward machen
    hand_side = project_onto_plane(
        hand_side_raw,
        hand_forward
    )

    hand_side = unit(
        hand_side
    )

    if hand_side is None:
        return None


# ----------------------------------------
# NORMALENVEKTOR DER HANDFLÄCHE
# ----------------------------------------

    hand_normal = unit(
        np.cross(
            hand_side,
            hand_forward
        )
    )

    if hand_normal is None:
        return None


    # ========================================
    # REFERENZEBENE DES ELLBOGENS
    # ========================================

    elbow_plane_cross = np.cross(
        upper_arm,
        forearm_axis
    )

    elbow_plane_length = np.linalg.norm(
        elbow_plane_cross
    )


    # Wenn der Arm fast gerade ist,
    # lässt sich die Ebene schlecht bestimmen.
    if elbow_plane_length < 0.20:

        return {
            "forearm_rotation_deg": None
        }


    elbow_plane_normal = unit(
        elbow_plane_cross
    )


    # ========================================
    # UNTERARMROTATION
    # ========================================

    forearm_rotation_deg = (
        signed_angle_about_axis(
            elbow_plane_normal,
            hand_side,
            forearm_axis
        )
    )

    # ========================================
# HANDGELENK
# ========================================

# Wir vergleichen die Richtung des Unterarms
# mit der Längsrichtung der Hand.
#
# Als Drehachse verwenden wir die Querachse
# der Hand (kleiner Finger -> Zeigefinger).

    wrist_bend_deg = (
        signed_angle_about_axis(
            forearm_axis,
            hand_forward,
            hand_side
        )
    )

    # ========================================
    # ZEIGEFINGER
    # ========================================

    # ----------------------------------------
    # PROXIMALE BEUGUNG
    # ----------------------------------------
    #
    # Vergleicht die Richtung der Handfläche
    # mit dem ersten Fingerglied.
    #
    # Gerade:
    # hand_forward und MCP->PIP ungefähr parallel
    # => etwa 0°
    #
    # Finger gebeugt:
    # => Winkel wird größer

    index_first_segment = unit(
        index_pip -
        index_mcp
    )

    if index_first_segment is None:

        index_proximal_deg = None

    else:

        index_proximal_deg = angle_between(
            hand_forward,
            index_first_segment
        )


    # ----------------------------------------
    # PIP-BEUGUNG
    # ----------------------------------------

    index_pip_inner = angle_between(
        index_mcp - index_pip,
        index_dip - index_pip
    )

    if index_pip_inner is None:

        index_pip_bend_deg = None

    else:

        index_pip_bend_deg = (
            180.0 -
            index_pip_inner
        )


    # ----------------------------------------
    # DIP-BEUGUNG
    # ----------------------------------------

    index_dip_inner = angle_between(
        index_pip - index_dip,
        index_tip - index_dip
    )

    if index_dip_inner is None:

        index_dip_bend_deg = None

    else:

        index_dip_bend_deg = (
            180.0 -
            index_dip_inner
        )


    # ----------------------------------------
    # DISTALE ROBOTERBEUGUNG
    # ----------------------------------------
    #
    # pib hat nur ein distales Fingergelenk.
    # Deshalb kombinieren wir PIP + DIP.

    if (
        index_pip_bend_deg is None
        or
        index_dip_bend_deg is None
    ):

        index_distal_deg = None

    else:

        index_distal_deg = (
            0.65 * index_pip_bend_deg
            +
            0.35 * index_dip_bend_deg
        )

    return {

        "forearm_rotation_deg":
            forearm_rotation_deg,

        "wrist_bend_deg":
            wrist_bend_deg,

        "index_proximal_deg":
            index_proximal_deg,

        "index_distal_deg":
            index_distal_deg
    }

def draw_hand(
    frame,
    hand_landmarks
):

    height, width = frame.shape[:2]


    def point(index):

        landmark = hand_landmarks[
            index
        ]

        return (
            int(
                landmark.x * width
            ),
            int(
                landmark.y * height
            )
        )


    # Verbindungen
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            point(start),
            point(end),
            (255, 255, 0),
            2
        )


    # Alle 21 Punkte
    for index in range(21):

        cv2.circle(
            frame,
            point(index),
            3,
            (0, 255, 255),
            -1
        )

def find_right_hand_index(
    hand_result,
    pose_landmarks
):
    """
    Sucht die erkannte Hand, deren Handgelenk
    am nächsten am rechten Pose-Handgelenk liegt.

    Dadurch müssen wir uns nicht auf
    Left/Right-Handedness verlassen.
    """

    if not hand_result.hand_landmarks:
        return None


    pose_wrist = pose_landmarks[
        RIGHT_WRIST
    ]


    best_index = None
    best_distance = float("inf")


    for i, hand in enumerate(
        hand_result.hand_landmarks
    ):

        hand_wrist = hand[
            HAND_WRIST
        ]


        dx = (
            hand_wrist.x
            -
            pose_wrist.x
        )

        dy = (
            hand_wrist.y
            -
            pose_wrist.y
        )


        distance = (
            dx * dx
            +
            dy * dy
        )


        if distance < best_distance:

            best_distance = distance
            best_index = i

    if best_distance > 0.03:
        return None

    return best_index


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

def ensure_hand_model():

    if HAND_MODEL_PATH.exists():
        return

    print(
        "MediaPipe-Handmodell wird "
        "einmalig heruntergeladen ..."
    )

    urllib.request.urlretrieve(
        HAND_MODEL_URL,
        HAND_MODEL_PATH
    )

    print("Handmodell gespeichert:")
    print(HAND_MODEL_PATH)


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

def angle_difference_deg(angle, reference):
    """
    Kürzeste Winkeldifferenz zwischen
    zwei Winkeln.

    Ergebnis:
    -180° bis +180°
    """

    return (
        (angle - reference + 180.0)
        % 360.0
        - 180.0
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

# ==========================================
# STABILES KÖRPER-KOORDINATENSYSTEM
# ==========================================

    body_right = unit(
        right_shoulder -
        left_shoulder
    )

    body_down_raw = unit(
        middle_hips -
        middle_shoulders
    )

    if body_right is None or body_down_raw is None:
        return None


    body_forward = unit(
        np.cross(
            body_right,
            body_down_raw
        )
    )

    if body_forward is None:
        return None


    # body_down neu berechnen,
    # damit alle drei Achsen rechtwinklig sind
    body_down = unit(
        np.cross(
            body_forward,
            body_right
        )
    )

    if body_down is None:
        return None

    # Wir berechnen hier den Weg von der Schulter zum Ellbogen in Vektorenform, um die Winkel zu berechnen
    upper_arm = unit(
        right_elbow -
        right_shoulder
    )

    if body_forward is None or upper_arm is None:
        return None


# ==========================================
# UNTERARM
# ==========================================

    forearm = unit(
        right_wrist -
        right_elbow
    )

    if forearm is None:
        return None


    # ==========================================
    # OBERARMROTATION
    # ==========================================

    elbow_cross = np.cross(
        upper_arm,
        forearm
    )

    elbow_cross_length = np.linalg.norm(
        elbow_cross
    )


    # Bei fast geradem Arm ist die
    # Oberarmrotation nicht zuverlässig messbar.

    if elbow_cross_length < 0.20:

        upper_arm_rotation_deg = None

    else:

        elbow_plane_normal = unit(
            elbow_cross
        )


        reference = project_onto_plane(
            body_forward,
            upper_arm
        )

        if reference is None:

            upper_arm_rotation_deg = None

        elif np.linalg.norm(reference) < 0.20:

            # Oberarm liegt nahezu parallel
            # zur Vorwärtsachse des Körpers.
            # Rotation ist hier instabil.
            upper_arm_rotation_deg = None

        else:

            reference = unit(
                reference
            )

            upper_arm_rotation_deg = (
                signed_angle_about_axis(
                    reference,
                    elbow_plane_normal,
                    upper_arm
                )
            )

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

    # ==========================================
    # SCHULTER: EBENE + HEBUNG
    # ==========================================

    # Anteil des Oberarms, der nicht nach unten
    # oder oben zeigt.
    horizontal_component = math.sqrt(
        side_component ** 2
        +
        front_component ** 2
    )


    # ------------------------------------------
    # 1. HEBUNG DES ARMS
    #
    # 0°   = Arm hängt nach unten
    # 90°  = Arm ungefähr waagerecht
    # 180° = Arm zeigt nach oben
    # ------------------------------------------

    shoulder_lift_deg = math.degrees(
        math.atan2(
            horizontal_component,
            down_component
        )
    )


    # ------------------------------------------
    # 2. BEWEGUNGSEBENE DER SCHULTER
    #
    # ungefähr:
    #
    # 0°   = Arm seitlich
    # +90° = Arm nach vorne
    # -90° = Arm nach hinten
    #
    # Wenn der Arm fast exakt nach oben/unten
    # zeigt, ist diese Ebene nicht eindeutig.
    # ------------------------------------------

    if horizontal_component < 0.15:

        shoulder_plane_deg = None

    else:

        shoulder_plane_deg = math.degrees(
            math.atan2(
                front_component,
                side_component
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
            ),

        "upper_arm_rotation_deg":
            upper_arm_rotation_deg,

        "shoulder_plane_deg":
            shoulder_plane_deg,

        "shoulder_lift_deg":
            float(
                np.clip(
                    shoulder_lift_deg,
                    0,
                    180
                )
            )
    }


def smooth_angles(previous, current):

    # Beim ersten gültigen Frame
    if previous is None:

        result = {}

        for key, value in current.items():

            if value is not None:
                result[key] = value

        return result


    result = previous.copy()


    for key, value in current.items():

        # Ungültigen Messwert ignorieren
        if value is None:
            continue


        # Neuer Winkel, den wir vorher
        # noch nicht hatten
        if key not in previous:

            result[key] = value
            continue


        # =====================================
        # PASSENDE GLÄTTUNG AUSWÄHLEN
        # =====================================

        if key in (
            "index_proximal_deg",
            "index_distal_deg"
        ):

            alpha = FINGER_SMOOTHING_ALPHA


        elif key in (
            "forearm_rotation_deg",
            "wrist_bend_deg"
        ):

            alpha = HAND_SMOOTHING_ALPHA


        else:

            alpha = SMOOTHING_ALPHA

        # =====================================
        # AUSREISSER BEI HANDWERTEN IGNORIEREN
        # =====================================

        if key == "forearm_rotation_deg":

            jump = angle_difference_deg(
                value,
                previous[key]
            )

            # Ein Sprung über 35° innerhalb
            # eines Frames ist wahrscheinlich
            # ein Trackingfehler.
            if abs(jump) > 90.0:
                continue


        if key == "wrist_bend_deg":

            jump = (
                value -
                previous[key]
            )

            if abs(jump) > 35.0:
                continue


        # =====================================
        # ZYKLISCHE WINKEL
        # =====================================

        if key in (
            "upper_arm_rotation_deg",
            "shoulder_plane_deg",
            "forearm_rotation_deg"
        ):

            delta = angle_difference_deg(
                value,
                previous[key]
            )


            # Unterarm zusätzlich etwas begrenzen
            if key == "forearm_rotation_deg":

                delta = max(
                    -25.0,
                    min(
                        25.0,
                        delta
                    )
                )


            result[key] = (
                previous[key]
                +
                alpha * delta
            )


            # wieder -180 ... +180
            result[key] = (
                (result[key] + 180.0)
                % 360.0
                - 180.0
            )


        # =====================================
        # NORMALE WINKEL
        # =====================================

        else:

            result[key] = (
                (1.0 - alpha)
                * previous[key]
                +
                alpha
                * value
            )


    return result

def project_onto_plane(v, normal):
    """
    Entfernt aus v den Anteil,
    der in Richtung normal zeigt.
    """

    normal = unit(normal)

    if normal is None:
        return None

    return (
        v
        -
        np.dot(v, normal)
        * normal
    )


def signed_angle_about_axis(a, b, axis):
    """
    Vorzeichenbehafteter Winkel von a nach b
    um die Achse axis.

    Ergebnis:
    -180° bis +180°
    """

    axis = unit(axis)

    if axis is None:
        return None


    a = project_onto_plane(
        a,
        axis
    )

    b = project_onto_plane(
        b,
        axis
    )


    if a is None or b is None:
        return None


    a = unit(a)
    b = unit(b)


    if a is None or b is None:
        return None


    cos_angle = np.clip(
        np.dot(a, b),
        -1.0,
        1.0
    )


    sin_angle = np.dot(
        axis,
        np.cross(a, b)
    )


    return math.degrees(
        math.atan2(
            sin_angle,
            cos_angle
        )
    )


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
    ensure_hand_model()

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

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
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

    hand_options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                HAND_MODEL_PATH
            )
        ),

        running_mode=
            RunningMode.VIDEO,

        # Beide Hände erkennen
        num_hands=2,

        min_hand_detection_confidence=0.6,

        min_hand_presence_confidence=0.6,

        min_tracking_confidence=0.7
    )

    filtered = None


    with (
        PoseLandmarker.create_from_options(
            options
        ) as pose_landmarker,

        HandLandmarker.create_from_options(
            hand_options
        ) as hand_landmarker):


        while True:

            raw_forearm_rotation = None
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
                pose_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            hand_result = (
                hand_landmarker.detect_for_video(
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

                right_hand_index = (
                    find_right_hand_index(
                        hand_result,
                        image_landmarks
                    )
                )


                right_hand_landmarks = None
                right_hand_world = None


                if right_hand_index is not None:

                    right_hand_landmarks = (
                        hand_result.hand_landmarks[
                            right_hand_index
                        ]
                    )

                    right_hand_world = (
                        hand_result.hand_world_landmarks[
                            right_hand_index
                        ]
                    )


                    draw_hand(
                        frame,
                        right_hand_landmarks
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

                    if (
                        angles is not None
                        and
                        right_hand_world is not None
                    ):

                        hand_angles = (
                            calculate_right_hand_angles(
                                world_landmarks,
                                right_hand_world
                            )
                        )

                        if hand_angles is not None:

                            raw_forearm_rotation = (
                                hand_angles.get(
                                    "forearm_rotation_deg"
                                )
                            )

                            angles.update(
                                hand_angles
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

                        if angles is not None:

                            # =====================================
                            # WINKEL GLÄTTEN – bereits vorhanden
                            # =====================================

                            filtered = smooth_angles(
                                filtered,
                                angles
                            )

                            # =====================================
                            # NEU: ZEIGEFINGERWERTE AUSLESEN
                            # =====================================

                            index_proximal = filtered.get(
                                "index_proximal_deg"
                            )

                            index_distal = filtered.get(
                                "index_distal_deg"
                            )

                            # =====================================
                            # NEU: TEXTE FÜR DIE ANZEIGE ERZEUGEN
                            # =====================================

                            if index_proximal is None:
                                index_proximal_text = "?"
                            else:
                                index_proximal_text = f"{index_proximal:.0f}"

                            if index_distal is None:
                                index_distal_text = "?"
                            else:
                                index_distal_text = f"{index_distal:.0f}"

                            # =====================================
                            # DEIN BISHERIGER CODE GEHT HIER WEITER
                            # =====================================

                            rotation = filtered.get(
                                "upper_arm_rotation_deg"
                            )

                            plane = filtered.get(
                                "shoulder_plane_deg"
                            )

                            lift = filtered.get(
                                "shoulder_lift_deg"
                            )



                        filtered_forearm_rotation = (
                            filtered.get(
                                "forearm_rotation_deg"
                            )
                        )

                        rotation = filtered.get(
                            "upper_arm_rotation_deg"
                        )

                        plane = filtered.get(
                            "shoulder_plane_deg"
                        )

                        lift = filtered.get(
                            "shoulder_lift_deg"
                        )

                        forearm_rotation = filtered.get(
                            "forearm_rotation_deg"
                        )

                        if (
                            raw_forearm_rotation is not None
                            and
                            filtered_forearm_rotation is not None
                        ):

                            print(
                                f"RAW: {raw_forearm_rotation:+7.1f} | "
                                f"FILTER: {filtered_forearm_rotation:+7.1f}"
                            )

                        wrist_bend = filtered.get(
                            "wrist_bend_deg"
                        )


                        if wrist_bend is None:

                            wrist_text = "nicht messbar"

                        else:

                            wrist_text = (
                                f"{wrist_bend:+.0f} deg"
                            )

                        if forearm_rotation is None:

                            forearm_text = "nicht messbar"

                        else:

                            forearm_text = (
                                f"{forearm_rotation:+.0f} deg"
                            )


                        if plane is None:
                            plane_text = "nicht messbar"
                        else:
                            plane_text = f"{plane:+.0f} deg"


                        if lift is None:
                            lift_text = "nicht messbar"
                        else:
                            lift_text = f"{lift:.0f} deg"

                        if rotation is None:

                            rotation_text = "nicht messbar"

                        else:

                            rotation_text = (
                                f"{rotation:+.0f} deg"
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


                                #  f"Eb "
                                #  f"{plane_text} | "

                                #  f"Heb "
                                #  f"{lift_text} | "

                                #  f"Ell "
                                #  f"{filtered['elbow_bend_deg']:.0f} deg | "

                                # f"ObRot "
                                # f"{rotation_text}"

                                
                                # f"UntRot "
                                # f"{forearm_text}"

                         
                                # f"Handgelenk "
                                # f"{wrist_text}"


                                f"IndexProx {index_proximal_text} deg | "
                                f"IndexDist {index_distal_text} deg"

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