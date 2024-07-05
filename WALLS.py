import numpy as np

WALLS = {

        'EmptyRoom':
                np.array([
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                ]),

        'EmptyRoomBounds':
                np.array([
                    # x1, y1, x2, y2
                    (0, 0, 17, 1),     # Left wall
                    (0, 16, 17, 17),   # Right wall
                    (0, 0, 1, 17),     # Top wall
                    (16, 0, 17, 17),   # Bottom wall
                ]),

        'FourRooms':
                np.array([
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], ## tunnel
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1], 
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1], ## tunnel
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], ## tunnel
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                ]),

        'FourRoomsBounds':
                np.array([
                    # x1, y1, x2, y2
                    (0, 0, 17, 1),     # Left wall
                    (0, 16, 17, 17),   # Right wall
                    (0, 0, 1, 17),     # Top wall
                    (16, 0, 17, 17),   # Bottom wall
                    
                    (1, 8, 4, 9),     # middle up wall
                    (5, 8, 12, 9),     # middle mid wall
                    (13, 8, 16, 9),     # middle down wall

                    (8, 1, 9, 4),       # middle left wall
                    (8, 5, 9, 12),      # middle mid wall
                    (8, 13, 9, 16),     # middle right wall
                ]),
}