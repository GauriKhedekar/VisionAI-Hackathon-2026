import pygame
import cv2
import mediapipe as mp
from vision_agents.core.state import ObjectState

# -------------------- INIT --------------------
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vision AI - Gesture Controlled Engine")

clock = pygame.time.Clock()

obj = ObjectState()
obj.convert_to_3d("sphere")

position = [WIDTH // 2, HEIGHT // 2, 0]

RADIUS = 20
FORCE = 0.4

# -------------------- OpenCV Setup --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

running = True

# -------------------- MAIN LOOP --------------------
while running:
    screen.fill((20, 20, 30))

    # --- Pygame exit ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Camera frame ---
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Reset acceleration
    obj.apply_force([0.0, 0.0, 0.0])

    # --- Gesture Detection ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = hand_landmarks.landmark[0].x
            y = hand_landmarks.landmark[0].y

            # Horizontal control
            if x > 0.7:
                obj.apply_force([FORCE, 0, 0])
            elif x < 0.3:
                obj.apply_force([-FORCE, 0, 0])

            # Vertical control
            if y > 0.7:
                obj.apply_force([0, FORCE, 0])
            elif y < 0.3:
                obj.apply_force([0, -FORCE, 0])

    # --- Physics Update ---
    obj.update_physics()
    position = obj.update_position(position)

    # --- Boundary Collision ---
    if position[0] <= RADIUS or position[0] >= WIDTH - RADIUS:
        obj.velocity[0] *= -1

    if position[1] <= RADIUS or position[1] >= HEIGHT - RADIUS:
        obj.velocity[1] *= -1

    position[0] = max(RADIUS, min(WIDTH - RADIUS, position[0]))
    position[1] = max(RADIUS, min(HEIGHT - RADIUS, position[1]))

    # --- Draw Object ---
    rect_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
    rect_surface.fill((0, 200, 255))

    rotated = pygame.transform.rotate(rect_surface, obj.rotation)
    rect_rect = rotated.get_rect(center=(int(position[0]), int(position[1])))

    screen.blit(rotated, rect_rect)

    pygame.display.flip()
    clock.tick(60)

    # Show camera window
    cv2.imshow("Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
pygame.quit()