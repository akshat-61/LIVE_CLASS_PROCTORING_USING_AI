from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=40,
            n_init=2,
            max_cosine_distance=0.2
        )

    def update(self, detections, frame):
        """
        detections: [[x1, y1, x2, y2, confidence]]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            track_id = track.track_id

            results.append({
                "id": f"S{track_id:03}",
                "bbox": [int(l), int(t), int(r), int(b)]
            })

        return results
