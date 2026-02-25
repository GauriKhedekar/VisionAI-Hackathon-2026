class ObjectState:
    def __init__(self):
        self.type = None
        self.dimension = "2D"

        # Physics
        self.velocity = [0.0, 0.0, 0.0]
        self.acceleration = [0.0, 0.0, 0.0]

        # Rotation
        self.rotation = 0.0
        self.rotation_speed = 0.0

    def convert_to_3d(self, new_type):
        self.type = new_type
        self.dimension = "3D"

    # Apply continuous force (acceleration)
    def apply_force(self, force_vector):
        self.acceleration = force_vector

    # Update physics each frame
    def update_physics(self):
        # v = v + a
        self.velocity = [
            self.velocity[i] + self.acceleration[i]
            for i in range(3)
        ]

        # rotation update
        self.rotation += self.rotation_speed

    def update_position(self, position):
        # x = x + v
        return [
            position[i] + self.velocity[i]
            for i in range(3)
        ]

    def stop(self):
        self.velocity = [0.0, 0.0, 0.0]
        self.acceleration = [0.0, 0.0, 0.0]