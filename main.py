#!/usr/bin/env python3

import cv2
from cvzone.HandTrackingModule import HandDetector
from typing import NamedTuple

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(detectionCon=0.8, maxHands=2)

class Vector2(NamedTuple):
    x: int
    y: int

class Rectangle(NamedTuple):
    pos: Vector2
    size: Vector2

    @property
    def bottom_right(self):
        return Vector2(self.pos.x + self.size.x, self.pos.y + self.size.y)

    @property
    def top_left(self):
        return self.pos

    @property
    def center(self):
        return Vector2(self.pos.x + (self.size.x // 2), self.pos.y + (self.size.y // 2))

    def contains(self, point):
        return self.pos.x <= point[0] <= self.pos.x + self.size.x and self.pos.y <= point[1] <= self.pos.y + self.size.y

    @staticmethod
    def from_center(center: Vector2, size: Vector2):
        return Rectangle(
            Vector2(center.x - (size.x // 2), center.y - (size.y // 2)),
            size,
        )

INDEX_FINGER = 8
MIDDLE_FINGER = 12

r = Rectangle(Vector2(100, 100), Vector2(200, 200))

while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        h = hands[0]

        lm, bbox = h['lmList'], h['bbox']

        index_finger = lm[INDEX_FINGER]
        index_finger = Vector2(index_finger[0], index_finger[1])

        middle_finger = lm[MIDDLE_FINGER]
        middle_finger = Vector2(middle_finger[0], middle_finger[1])

        finger_dist, _, _ = detector.findDistance(index_finger, middle_finger, img)

        enable_movement = finger_dist < 50

        color = (255, 0, 255)
        fill_mode = cv2.FILLED
        if enable_movement:
            if r.contains(index_finger):
                color = (255, 0, 0)
                fill_mode = cv2.LINE_4

                r = Rectangle.from_center(index_finger, r.size)

        cv2.rectangle(img, r.top_left, r.bottom_right, color, fill_mode)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
