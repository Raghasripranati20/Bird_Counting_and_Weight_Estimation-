import cv2

def draw_annotations(frame, tracks, count):
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track.id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    cv2.putText(frame, f"Bird Count: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return frame
