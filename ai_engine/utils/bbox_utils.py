def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0

    return interArea / float(boxAArea + boxBArea - interArea)


def assign_object_to_person(persons, objects):
    mapping = {}

    for person in persons:
        best_iou = 0
        assigned_obj = None

        for obj in objects:
            score = iou(person["bbox"], obj["bbox"])
            if score > best_iou:
                best_iou = score
                assigned_obj = obj

        if best_iou > 0.3:
            mapping[person["id"]] = assigned_obj

    return mapping
