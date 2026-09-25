from controller import Robot

import json
import math
import socket


# ============================================
# UDP
# ============================================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005


# ============================================
# KALIBRIERUNG
# ============================================

CALIBRATION = {

    # ========================================
    # SCHULTEREBENE
    # ========================================

    "shoulder_vertical_right": {

        "source":
            "shoulder_plane_deg",

        # Wird gleich gemessen!
        "source_neutral_deg":
            10.0,

        # Ebene ist ein zyklischer Winkel
        "cyclic":
            True,

        "sign":
            1.0,

        "scale":
            0.7,

        "offset_deg":
            -90.0,

        "max_delta_deg":
            90.0
    },


    # ========================================
    # ARMANHEBUNG
    # ========================================

    "shoulder_horizontal_right": {

        "source":
            "shoulder_lift_deg",

        # Wird gleich gemessen!
        "source_neutral_deg":
            103.0,

        "sign":
            -1.0,

        "scale":
            1.0,

        "offset_deg":
            0.0,

        "max_delta_deg":
            90.0
    },


    "elbow_right": {
        "source": "elbow_bend_deg",

        # Diesen Wert messen wir gleich
        "source_neutral_deg": 70.0,

        "sign": 1.0,
        "scale": 1.0,

        # Erst einmal vorsichtig
        "offset_deg": 30.0,

        "max_delta_deg": 60.0
    },

    "upper_arm_right": {

        "source":
            "upper_arm_rotation_deg",

        "source_neutral_deg":
            155.0,  # durch deinen echten Neutralwert ersetzen

        "cyclic":
            True,

        "sign":
            1.0,

        "scale":
            1.0,

        "offset_deg":
           -90.0,

        "max_delta_deg":
            45.0
    },

    "forearm_right": {

        "source":
            "forearm_rotation_deg",

        # Deinen echten Neutralwert einsetzen
        "source_neutral_deg":
            -100.0,

        # Wichtig wegen -180° / +180°
        "cyclic":
            True,

        "sign":
            -1.0,

        # Erst vorsichtig
        "scale":
            1.0,

        # pib-Neutralposition
        "offset_deg":
            0.0,

        "max_delta_deg":
            60.0
    },

    "wrist_right": {

        "source":
            "wrist_bend_deg",

        # Dein gemessener Wert
        "source_neutral_deg":
            0.0,

        "sign":
            -1.0,

        "scale":
            3.0,

        "offset_deg":
            0.0,

        "max_delta_deg":
            90.0
    },

    "index_right_proximal": {

        "source": "index_proximal_deg",

        # Noch nicht kalibriert:
        # Hier kommt dein Kamerawert bei geradem Finger hinein.
        "source_neutral_deg": 13.0,

        # Drehrichtung nach dem ersten Test bestätigen.
        "sign": 1.0,

        # Zunächst kleiner Bewegungsumfang.
        "scale": 1.0,

        # Hier kommt der Webots-Winkel für den offenen Finger hinein.
        "offset_deg": 0.0,

        "max_delta_deg": 45.0
    },

    "index_right_distal": {

        "source": "index_distal_deg",

        "source_neutral_deg": 20.0,

        "sign": 1.0,

        "scale": 1.0,

        "offset_deg": 0.0,

        "max_delta_deg": 45.0
    },

}

def angle_difference_deg(angle, reference):
    """
    Berechnet die kürzeste Differenz
    zwischen zwei Winkeln.

    Ergebnis:
    -180° bis +180°
    """

    return (
        (angle - reference + 180.0)
        % 360.0
        - 180.0
    )

def clamp_to_motor_limits(
    motor,
    target
):

    min_position = (
        motor.getMinPosition()
    )

    max_position = (
        motor.getMaxPosition()
    )


    # Wenn beide 0 sind,
    # sind in Webots normalerweise
    # keine Soft-Limits aktiv.

    if (
        min_position == 0.0
        and
        max_position == 0.0
    ):

        return target


    return max(

        min_position,

        min(
            max_position,
            target
        )
    )


def main():

    robot = Robot()


    timestep = int(
        robot.getBasicTimeStep()
    )


    # ========================================
    # MOTOREN HOLEN
    # ========================================

    motors = {}


    for name in CALIBRATION:


        motor = robot.getDevice(
            name
        )


        motors[name] = motor


        max_velocity = (
            motor.getMaxVelocity()
        )


        # Bewegung etwas langsamer machen

        if max_velocity > 0:

            motor.setVelocity(
                min(
                    1.5,
                    max_velocity
                )
            )


        print(

            f"{name}: "

            f"min="
            f"{motor.getMinPosition():.3f}, "

            f"max="
            f"{motor.getMaxPosition():.3f}, "

            f"maxVel="
            f"{max_velocity:.3f}"
        )


    # ========================================
    # UDP EMPFÄNGER
    # ========================================

    sock = socket.socket(

        socket.AF_INET,
        socket.SOCK_DGRAM
    )


    sock.bind(
        (
            UDP_IP,
            UDP_PORT
        )
    )


    sock.setblocking(
        False
    )


    print(
        f"Warte auf Pose-Daten "
        f"auf UDP "
        f"{UDP_IP}:{UDP_PORT} ..."
    )


    latest = None


    # ========================================
    # WEBOTS HAUPTSCHLEIFE
    # ========================================

    while (
        robot.step(timestep)
        != -1
    ):


        # Alle Pakete lesen.
        # Uns interessiert nur
        # das jeweils neueste.

        while True:

            try:

                raw, address = (
                    sock.recvfrom(
                        4096
                    )
                )


                latest = json.loads(

                    raw.decode(
                        "utf-8"
                    )
                )


            except BlockingIOError:

                break


            except (
                UnicodeDecodeError,
                json.JSONDecodeError
            ) as error:

                print(
                    "Ungültiges UDP-Paket:",
                    error
                )

                break


        # Noch keine Kameradaten?

        if latest is None:
            continue


        # ====================================
        # MOTOREN BEWEGEN
        # ====================================

        for (
            motor_name,
            config
        ) in CALIBRATION.items():


            source = (
                config["source"]
            )


            if source not in latest:
                continue


            human_deg = float(
                latest[source]
            )


            # ==========================================
            # NEUTRALSTELLUNG DES MENSCHEN
            # ==========================================

            if "source_neutral_deg" in config:

                neutral_deg = config[
                    "source_neutral_deg"
                ]

                # Für zyklische Winkel wie die
                # Oberarmrotation
                if config.get(
                    "cyclic",
                    False
                ):

                    human_delta_deg = (
                        angle_difference_deg(
                            human_deg,
                            neutral_deg
                        )
                    )

                else:

                    human_delta_deg = (
                        human_deg
                        -
                        neutral_deg
                    )

            else:

                # Gelenke ohne spezielle Neutralstellung
                # funktionieren wie bisher.
                human_delta_deg = human_deg


            # ==========================================
            # MENSCHLICHE BEWEGUNG BEGRENZEN
            # ==========================================

            if "max_delta_deg" in config:

                max_delta = config[
                    "max_delta_deg"
                ]

                human_delta_deg = max(
                    -max_delta,
                    min(
                        max_delta,
                        human_delta_deg
                    )
                )


            # ==========================================
            # AUF ROBOTERWINKEL ABBILDEN
            # ==========================================

            robot_deg = (

                config["offset_deg"]

                +

                config["sign"]
                * config["scale"]
                * human_delta_deg
            )


            # Webots erwartet bei
            # Rotationsmotoren Radiant.

            target_rad = math.radians(
                robot_deg
            )


            motor = motors[
                motor_name
            ]


            target_rad = (
                clamp_to_motor_limits(
                    motor,
                    target_rad
                )
            )


            motor.setPosition(
                target_rad
            )


    sock.close()


if __name__ == "__main__":
    main()