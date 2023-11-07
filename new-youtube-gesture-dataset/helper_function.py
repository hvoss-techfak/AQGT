import bz2
import os
import pickle
import traceback

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

entity_dict = {}


def init_entity_dict():
    global entity_dict
    entity_dict["Hufeisen"] = ["hufeisen"]
    entity_dict["Stockwerk"] = ["stockwerk", "stock", "geschoss", "etage", "etagen"]
    entity_dict["Treppe"] = ["treppe", "treppen", "notfalltreppen", "feuerleiter", "wendeltreppe", "leiter",
                             "feuertreppen", "wendeltreppen", "schlängeltreppen"]
    entity_dict["Fenster"] = ["fensterreihe", "fensterreihen", "glasfenster", "kirchenfenster", "fenster",
                              "rosette", "rundfenster", "fenster"]
    entity_dict["Kirche"] = ["kirchengebäude", "moscheee", "kapellen", "kirchenschiff", "schiffaufbau",
                             "kapelle", "kirchen", "kirche", "kirchtürme", "Schiffaufbau", "nebengängen",
                             "hauptschiff", "türmen", "längsschiff", "kirchtürmen", "moschee", "flügel"]
    entity_dict["Tür"] = ["tür", "türen", "eingang", "bogentür", "holztüren", "eingangstor", "tor",
                          "haupteingang"]
    entity_dict["Aussenobjekte"] = ["grünstück", "rasen", "laubbaum", "baum", "bäume", "bäumen", "uhr",
                                    "wasser", "hecken", "boden", "metallzaun", "wasserfläche", "stacheldraht"]
    entity_dict["Lampen"] = ["laternen", "laterne", "lampen", "leuchten", "straßenlaternen"]
    entity_dict["Turm"] = ["türme", "kircht", "kirchturm", "t", "türmchen", "zwillingsturm", "turm", "spitzen",
                           "aussichtsturm", "aussichtstürme"]
    entity_dict["Schale"] = ["kelchen", "kelch", "brunnenschalen", "auffangbecken", "schalen", "wannen",
                             "schüsseln", "untertasse", "tassen", "schale", "becken", "wanne", "servierschalen",
                             "teller", "brunnenschale", "teller"]
    entity_dict["Straße"] = ["straße", "t-kreuzung", "s-kurve"]
    entity_dict["Rathaus"] = ["rathaus"]
    entity_dict["Platz"] = ["platz", "innenhof", "sockel", "riesenplatz", "kirchplatz", "fläche", "plateau",
                            "podest"]
    entity_dict["Gebäude"] = ["gebäudeteil", "gebäude", "häuser", "häusern", "schloss", "rundbögen", "halle",
                              "balkon", "balkone", "außengebäuden", "gebaudeteil","gebäude"]
    entity_dict["Kunstobjekt"] = ["säule", "ding", "form", "betonsockel", "röhren", "roehren", "ring",
                                  "stangen", "betonsäulue", "betonsäule"]
    entity_dict["Dach"] = ["rundkuppel", "spitzdach", "vordach", "kuppel", "flachbau", "dach", "giebel",
                           "walldach", "giebel", "walmdach", "flachdach", "runddach", "kirchdach", "dächern",
                           "spitze"]
    entity_dict["Brunnen"] = ["brunnen"]
    entity_dict["Hecke"] = ["hecke"]
    # entity_dict["X"] = ["aufgänge", "glasvorbau", "enden", "vorbau", "rohr", "öffnung", "objekt", "einkerbung",
    #                     "frontbau", "ecken", "pyramide", "höhe", "radius", "außenteile", "glassvorbau",
    #                     "teilen", "flügeln", "rundbogen", "doppelkaskade", "kugelabschnitt", "einkerbung",
    #                     "kreuzaufbau", "erhebung", "seite", "Doppelkaskade", "klotz", "portal",
    #                     "Kugelabschnitt", "einkerbungen", "seiten", "bogen", "kanten", "stück", "rahmen",
    #                     "anfang", "front", "meter", "riesenteil", "vorderseite", "teil", "bereich", "block",
    #                     "kreuzform", "ende", "stahlbogen", "metallwand", "mauer", "platten"]

from collections import OrderedDict
position_dict = OrderedDict()

position_dict["links"] = ["links","linken","linke","link"]
position_dict["rechts"] = ["rechts","rechten","rechte","recht"]

position_dict["davor"] = ["davor","vor","aussen","außen","vorne","vorm"]
position_dict["dahinter"] = ["dahinter","hinter","hinten","hinterm","weiter","dahinten","hinteren"]

position_dict["drauf"] = ["auf","drüber","drauf","übereinenander","übereinander","übereienander","aufm","oben","obere","über","dadrüber","darauf","bereinander","ü bereinander"]
position_dict["drunter"] = ["unter","drunter","unten","untere"]

position_dict["drinnen"] = ["in","innen","ineinander","drin","drinnen"]
position_dict["zwischen"] = ["zwischen","durch"]
position_dict["daneben"] = ["daneben","bei","an","dran"]
position_dict["drumherum"] = ["um","drumherum","drum","rum"]

position_dict["zusammen"] = ["nebeneinander","zusammen","dieselben","anderen","miteinander","umgeben","zusammen","gleiche","dieselben"]

position_dict["hin"] = ["zu","hin","von","vom"] 
position_dict["weg"] = ["raus","heraus"] 

position_dict["hier"] = ["hier","da","dieser","diesem","diese","dort","diesen"] 




def getPositionFromWordList(wlist):
    global position_dict
    for k in position_dict.keys():
        for start,stop,word in wlist:
            if word.lower() in position_dict[k]:
                return k
    return None


def getEntityFromWordList(wlist):
    global entity_dict
    if len(entity_dict.keys()) == 0:
        init_entity_dict()
    for start,stop,word in wlist:
        for k in entity_dict.keys():
            if word.lower() in entity_dict[k]:
                return k
    return None



def getAngle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def getHandPositions(ee):
    timingAnnotation = annotationsToTimings(ee)
    LeftHandShape = insertATimings(list(ee.tiers["R.G.Left.HandShapeShape"][1].values()), timingAnnotation)
    # LeftPathHandShape = list(ee.tiers["R.G.Left.PathofHandShape"][1].values())
    # LeftHSMovementDirection = list(ee.tiers["R.G.Left.HSMovementDirection"][1].values())
    # LeftHandShapeMovementRepetition = list(ee.tiers["R.G.Left.HandShapeMovementRepetition"][1].values())
    LeftPalmDirection = insertATimings(list(ee.tiers["R.G.Left.PalmDirection"][1].values()), timingAnnotation)
    # LeftPathOfPalmDirection = list(ee.tiers["R.G.Left.PathOfPalmDirection"][1].values())
    # LeftPalmDirectionMovementDirection = list(ee.tiers["R.G.Left.PalmDirectionMovementDirection"][1].values())
    # LeftPalmDirectionMovementRepetition = list(ee.tiers["R.G.Left.PalmDirectionMovementRepetition"][1].values())
    LeftBackOfHandDirection = insertATimings(list(ee.tiers["R.G.Left.BackOfHandDirection"][1].values()),
                                             timingAnnotation)
    LeftBackOfHandDirectionMovement = insertATimings(list(ee.tiers["R.G.Left.BackOfHandDirectionMovement"][1].values()),
                                                     timingAnnotation)
    LeftWristPosition = insertATimings(list(ee.tiers["R.G.Left.WristPosition"][1].values()), timingAnnotation)
    LeftWristDistance = insertATimings(list(ee.tiers["R.G.Left.WristDistance"][1].values()), timingAnnotation)
    LeftPathOfWristLocation = insertATimings(list(ee.tiers["R.G.Left.PathOfWristLocation"][1].values()),
                                             timingAnnotation)
    LeftWristLocationMovementDirection = insertATimings(
        list(ee.tiers["R.G.Left.WristLocationMovementDirection"][1].values()), timingAnnotation)
    # LeftWristLocationMovementRepetition = list(ee.tiers["R.G.Left.WristMovementRepetition"][1].values())
    LeftExtent = insertATimings(list(ee.tiers["R.G.Left.Extent"][1].values()), timingAnnotation)
    # LeftTemporalSequence = list(ee.tiers["R.G.Left.TemporalSequence"][1].values() if "R.G.Left.TemporalSequence" in ee.tiers.keys() else [])
    LeftPractice = insertATimings(list(ee.tiers["R.G.Left.Practice"][1].values()), timingAnnotation)

    ctempLeft = [LeftHandShape,
                 LeftPalmDirection,
                 LeftBackOfHandDirection,
                 LeftBackOfHandDirectionMovement,
                 LeftWristPosition,
                 LeftWristDistance,
                 LeftPathOfWristLocation,
                 LeftWristLocationMovementDirection,
                 LeftExtent,
                 LeftPractice]
    # if ccountLeft is None:
    #     ccountLeft = []
    #     for i in range(12):
    #         ccountLeft.append([])
    # for i in range(len(ctemp)):
    #     ccountLeft[i] += [f[1] for f in ctemp[i]]

    RightHandShape = insertATimings(list(ee.tiers["R.G.Right.HandShapeShape"][1].values()), timingAnnotation)
    # RightPathHandShape = list(ee.tiers["R.G.Right.PathofHandShape"][1].values())
    # RightHSMovementDirection = list(ee.tiers["R.G.Right.HSMovementDirection"][1].values())
    # RightHandShapeMovementRepetition = list(ee.tiers["R.G.Right.HandShapeMovementRepetition"][1].values())
    RightPalmDirection = insertATimings(list(ee.tiers["R.G.Right.PalmDirection"][1].values()), timingAnnotation)
    # RightPathOfPalmDirection = list(ee.tiers["R.G.Right.PathOfPalmDirection"][1].values())
    # RightPalmDirectionMovementDirection = list(ee.tiers["R.G.Right.PalmDirectionMovementDirection"][1].values())
    # RightPalmDirectionMovementRepetition = list(ee.tiers["R.G.Right.PalmDirectionMovementRepetition"][1].values() if "R.G.Right.PalmDirectionMovementRepetition" in ee.tiers.keys() else ee.tiers["R.G.Right.PalmDirectionMovementRepetition.fhhr"][1].values())
    RightBackOfHandDirection = insertATimings(list(ee.tiers["R.G.Right.BackOfHandDirection"][1].values()),
                                              timingAnnotation)
    RightBackOfHandDirectionMovement = insertATimings(
        list(ee.tiers["R.G.Right.BackOfHandDirectionMovement"][1].values()), timingAnnotation)
    RightWristPosition = insertATimings(list(ee.tiers["R.G.Right.WristPosition"][1].values()), timingAnnotation)
    RightWristDistance = insertATimings(list(ee.tiers["R.G.Right.WristDistance"][1].values()), timingAnnotation)
    RightPathOfWristLocation = insertATimings(list(ee.tiers["R.G.Right.PathOfWristLocation"][1].values()),
                                              timingAnnotation)
    RightWristLocationMovementDirection = insertATimings(
        list(ee.tiers["R.G.Right.WristLocationMovementDirection"][1].values()), timingAnnotation)
    # RightWristLocationMovementRepetition = list(ee.tiers["R.G.Right.WristMovementRepetition"][1].values())
    RightExtent = insertATimings(list(ee.tiers["R.G.Right.Extent"][1].values()), timingAnnotation)
    # RightTemporalSequence = list(ee.tiers["R.G.Right.TemporalSequence"][1].values() if "R.G.Right.TemporalSequence" in ee.tiers.keys() else [])
    RightPractice = insertATimings(list(ee.tiers["R.G.Right.Practice"][1].values()), timingAnnotation)

    ctempRight = [RightHandShape,
                  RightPalmDirection,
                  RightBackOfHandDirection,
                  RightBackOfHandDirectionMovement,
                  RightWristPosition,
                  RightWristDistance,
                  RightPathOfWristLocation,
                  RightWristLocationMovementDirection,
                  RightExtent,
                  RightPractice]

    # if ccountRight is None:
    #     ccountRight = []
    #     for i in range(18):
    #         ccountRight.append([])
    # for i in range(len(ctemp)):
    #     ccountLeft[i] += [f[1] for f in ctemp[i]]

    # RS2 = list(ee.tiers["R.S.Semantic Feature" if "R.S.Semantic Feature" in ee.tiers.keys() else "R.S.Sematic Feature"][0].values())
    # RS = insertTimings(RS, timings)
    # RS2 = insertTimings(RS2, timings)
    return ctempLeft, ctempRight


def annotationsToTimings(ee):
    timings = ee.timeslots
    etiers = ee.tiers
    out = {}
    for k in etiers.keys():
        a, _, _, _ = etiers[k]
        for ak in a.keys():

            if not ak in out:
                out[ak] = (timings[a[ak][0]], timings[a[ak][1]])
            else:
                print(out[ak], (a[ak][0], a[ak][1]))
    return out


def getAnnotationwithTiming(ee, tier):
    timings = ee.timeslots
    LeftPhase = ee.tiers[tier][0]
    out = []
    for k in LeftPhase.keys():
        temp = LeftPhase[k]
        out.append((timings[temp[0]], timings[temp[1]], temp[2]))
    return out


def getPhases(ee):
    return getAnnotationwithTiming(ee, "R.G.Left.Phase"), getAnnotationwithTiming(ee, "R.G.Right.Phase")


def removeRotation(out):
    for i in range(3):
        out = out.copy()

        m1 = out[8].copy()
        out -= m1

        x = out[0]
        y = out[8]
        angle = getAngle(x[[1, 2]], y[[1, 2]], [0, 1]) - 90
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([1, 0, 0])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        out = rotation.apply(out)

        x = out[0]
        y = out[8]
        angle = -getAngle(x[[0, 1]], y[[0, 1]], [1, 0]) + 90
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        out = rotation.apply(out)
        out += m1

        m2 = out[14].copy()
        out -= m2
        x = out[14]
        y = out[11]
        angle = -getAngle(y[[0, 2]], x[[0, 2]], [0, 1]) + 90
        rotation_radians = np.radians(angle)
        rotation_axis = np.array([0, 1, 0])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        out = rotation.apply(out)
        out += m2
    return out


def getAveragePositions(features):
    ret = {}
    for i in range(len(features)):
        if features[i] is not None:
            e1 = features[i]
            for k in e1.keys():
                e2 = e1[k]
                avg = []
                if k in ret:
                    avg = ret[k]
                avg.append(e2["current_2d_pose"].mean(axis=0))
                ret[k] = avg
    for k in ret.keys():
        k1 = ret[k]
        k1 = np.asarray(k1).mean(axis=0)
        ret[k] = k1
    return ret


def checkscore(person, boneslist, threshold):
    for bone_idx in boneslist:
        if person[bone_idx] < threshold:
            return False
    return True


def toKeyPointList(features, ignoreBelow=30, scoreThreshold=0.2, stride=10,
                   boneslist=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19]):
    ret = {}
    seqL = {}
    lastidx = {}
    for i in range(len(features)):
        if features[i] is not None:
            f1 = features[i]
            for k in f1.keys():
                if not k in seqL.keys():
                    seqL[k] = []
                    lastidx[k] = i
                if lastidx[k] < i - 1:
                    seqL[k] = []
                    lastidx[k] = i

                if checkscore(f1[k]["current_2d_score"], boneslist, scoreThreshold):
                    seqL[k].append(f1["current_3d_pose"])
                if len(seqL[k]) >= ignoreBelow:
                    out = []
                    if k in ret.keys():
                        out = ret[k]
                    out.append(seqL[k].copy())
                    seqL[k] = seqL[k][stride:]


def checkPerc(file):
    try:
        feature = pickle.load(bz2.BZ2File(file, "rb"))
        ie = 0
        for i in range(len(feature)):
            if feature[i] is None:
                ie += 1
        frac = ie / len(feature)
        print(file, frac)
        if frac > 0.1:
            print("removing:", file)
            os.remove(file)
    except Exception:
        traceback.print_exc()
        os.remove(file)


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def reroute(d3_pose, d2_pose):
    d3_pose = d3_pose.copy()
    d2_pose = d2_pose.copy()
    d2_pose[:, 0] = ((d2_pose[:, 0] / 720) * 2) - 1
    d2_pose[:, 1] = ((d2_pose[:, 1] / 576) * 2) - 1

    d3_pose[0, :2] = d2_pose[19]
    d3_pose[1, :2] = d2_pose[11]
    d3_pose[2, :2] = d2_pose[13]
    d3_pose[3, :2] = d2_pose[15]
    d3_pose[4, :2] = d2_pose[12]
    d3_pose[5, :2] = d2_pose[14]
    d3_pose[6, :2] = d2_pose[16]
    d3_pose[7, :2] = (d2_pose[18] + d2_pose[19]) / 2
    d3_pose[8, :2] = d2_pose[18]
    d3_pose[9, :2] = d2_pose[0]
    d3_pose[10, :2] = d2_pose[17]
    d3_pose[11, :2] = d2_pose[5]
    d3_pose[12, :2] = d2_pose[7]
    d3_pose[13, :2] = d2_pose[9]
    d3_pose[14, :2] = d2_pose[6]
    d3_pose[15, :2] = d2_pose[8]
    d3_pose[16, :2] = d2_pose[10]
    return d3_pose


def normalizeLengths(d_poses):
    d_poses -= d_poses[8]
    #x_len = (abs(d_poses[11, 0]) + abs(d_poses[14, 0])) / 2
    #d_poses[:, 0] /= x_len
    #d_poses[:, 0] *= 0.2

    y_len = abs(d_poses[0, 1])
    d_poses /= y_len
    d_poses *= 0.6

    #z_len = abs(d_poses[9, 2])
    #d_poses[:, 2] /= z_len
    #d_poses[:, 2] *= 0.1
    return d_poses


def getLongestSequence(seq):
    out = None
    seq_len = 0
    for k in seq.keys():
        seq_temp = 0
        for l in seq[k]:
            seq_temp += len(l)
        if seq_temp > seq_len:
            seq_len = seq_temp
            out = k
    return seq_len, out


def debone(d3_pose):
    skeleton_parents = np.asarray([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_l = hand_parents_l + 17
    hand_parents_r = hand_parents_r + 17 + 21
    skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)
    out = []
    for i in range(len(d3_pose)):
        parent = skeleton_parents[i]
        if parent != -1:
            parent_bone = d3_pose[parent]
            new_bone = d3_pose[i] - parent_bone
            out.append(new_bone)
        else:
            out.append(d3_pose[i])
    return np.asarray(out)


def rebone(d3_pose):
    skeleton_parents = np.asarray([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_l = hand_parents_l + 17
    hand_parents_r = hand_parents_r + 17 + 21
    skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)
    out = []
    for i in range(len(d3_pose)):
        parent = skeleton_parents[i]
        if parent != -1:
            parent_bone = d3_pose[parent]
            new_bone = d3_pose[i] + parent_bone
            d3_pose[i] = new_bone
            out.append(new_bone)
        else:
            out.append(d3_pose[i])
    return np.asarray(out)


def deboneGesture(d3_pose):
    skeleton_parentsL = np.asarray([-1, 0, 1])
    skeleton_parentsR = np.asarray([-1, 24, 25])
    hand_parents_l = np.asarray([-1, -1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_r = np.asarray([-1, -1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_l = hand_parents_l + 3
    hand_parents_r = hand_parents_r + 3 + 21 + 3
    skeleton_parents = np.concatenate((skeleton_parentsL, hand_parents_l, skeleton_parentsR, hand_parents_r), axis=0)
    out = []
    for i in range(len(d3_pose)):
        parent = skeleton_parents[i]
        if parent != -1:
            parent_bone = d3_pose[parent]
            new_bone = d3_pose[i] - parent_bone
            out.append(new_bone)
        else:
            out.append(d3_pose[i])
    return np.asarray(out)


def reboneGesture(d3_pose):
    skeleton_parentsL = np.asarray([-1, 0, 1])
    skeleton_parentsR = np.asarray([-1, 24, 25])
    hand_parents_l = np.asarray([-1, -1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_r = np.asarray([-1, -1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    hand_parents_l = hand_parents_l + 3
    hand_parents_r = hand_parents_r + 3 + 21 + 3
    skeleton_parents = np.concatenate((skeleton_parentsL, hand_parents_l, skeleton_parentsR, hand_parents_r), axis=0)
    out = []
    for i in range(len(d3_pose)):
        parent = skeleton_parents[i]
        if parent != -1:
            parent_bone = d3_pose[parent]
            new_bone = d3_pose[i] + parent_bone
            d3_pose[i] = new_bone
            out.append(new_bone)
        else:
            out.append(d3_pose[i])
    return np.asarray(out)


def onlyGestures(d3Pose):
    outL = []
    outL.append(d3Pose[11])
    outL.append(d3Pose[12])
    outL.append(d3Pose[13])
    outL = np.asarray(outL)
    outL = np.concatenate((outL, d3Pose[17:38]), axis=0)
    outL -= outL[0]

    outR = []
    outR.append(d3Pose[14])
    outR.append(d3Pose[15])
    outR.append(d3Pose[16])
    outR = np.asarray(outR)
    outR = np.concatenate((outR, d3Pose[38:]), axis=0)
    outR -= outR[0]

    return np.concatenate((outL, outR), axis=0)



def flipRightGestures(d3Pose):
    d3Pose[24:, 0] = (1 - d3Pose[24:, 0]) - 1
    return d3Pose


def getPhase(leftPhases, rightPhases, cur_T):
    ret1 = None
    ret2 = None
    for (start, stop, type) in leftPhases:
        if start is not None and stop is not None and cur_T >= start and cur_T <= stop:
            type = type.replace("\n", "")
            ret1 = (start, stop, type)
            break
    for (start, stop, type) in rightPhases:
        if start is not None and stop is not None and cur_T >= start and cur_T <= stop:
            type = type.replace("\n", "")
            ret2 = (start, stop, type)
            break
    return ret1, ret2

def getAllAnnotationFromTime(anno,cur_t,extraTime=0):
    ret = []
    for start_t,stop_t,type in anno:
        if start_t is not None and stop_t is not None and start_t-extraTime <= cur_t <= stop_t+extraTime:
            ret.append((start_t,stop_t,type))
    return ret

def getAllAnnotationInRange(anno,anno_start,anno_stop,extraTime=0):
    ret = []
    for start_t,stop_t,type in anno:
        if start_t is not None and stop_t is not None:
            if start_t >= anno_start-extraTime and stop_t <= anno_stop+extraTime:
                ret.append((start_t,stop_t,type))
    return ret

def getAnnotationFromTime(anno,cur_t):
    for start_t,stop_t,type in anno:
        if start_t is not None and stop_t is not None and start_t <= cur_t <= stop_t:
            return (start_t,stop_t,type)




def getcurrentHands(leftHand, rightHand, cur_T):
    ret1 = [None for _ in range(10)]
    ret2 = [None for _ in range(10)]
    for i in range(10):
        te_L = leftHand[i]
        for (start, stop, type) in te_L:
            if start is not None and stop is not None and cur_T >= start and cur_T <= stop:
                ret1[i] = (start, stop, type)
                break
        te_R = rightHand[i]
        for (start, stop, type) in te_R:
            if start is not None and stop is not None and cur_T >= start and cur_T <= stop:
                ret2[i] = (start, stop, type)
                break
    return ret1, ret2

    # for (start,stop,type) in leftPhases:
    #     if start > cur_T and stop < cur_T:
    #         ret1 = (start,stop,type)
    #         break
    # for (start,stop,type) in rightPhases:
    #     if start > cur_T and stop < cur_T:
    #         ret2 = (start,stop,type)
    #         break
    # return ret1,ret2


def getLastXWords(RS, cur_t):
    small = 1000000000
    idx = 0
    for ie, (start, stop, word) in enumerate(RS):
        if start is not None and cur_t - start > 0 and cur_t - start < small:
            small = cur_t - start
            idx = ie
    ret = RS[:idx + 1][-100:]
    if len(ret) < 100:
        temp = []
        while len(ret) + len(temp) < 100:
            temp.append((0, 1, np.zeros((100,))))
        ret = temp + ret

    return ret


def lastXWordstoRelativeTime(words, cur_t, divider=10000):
    ret1, ret2, ret3 = [], [], []
    for (start, stop, word) in words:
        if start is not None and stop is not None:
            ret1.append(np.asarray((cur_t - start) / divider if not start == 0 else 0))
            ret2.append(np.asarray((cur_t - stop) / divider if not stop == 1 else 0))
            ret3.append(word)
    return ret1, ret2, ret3


def fillupList(seq, history_length):
    if len(seq) < history_length:
        temp = []
        while len(temp) + len(seq) < history_length:
            temp.append(np.zeros(seq[0].shape))
        seq = temp + seq
    return seq


def getTargetBonesFromSequence(seq, ie, idfinder, length=25):
    ret = []
    for i in range(ie + 1, ie + length + 1):
        seq_dict = seq[i]
        if seq_dict is not None and idfinder in seq_dict:
            poselist = seq_dict[idfinder]
            d3_pose = poselist["current_3d_pose"].copy()

            d3_pose = removeRotation(d3_pose)
            d3_pose = normalizeLengths(d3_pose)
            d3_pose = onlyGestures(d3_pose)

            d3_pose = deboneGesture(d3_pose.copy())
            ret.append(d3_pose)
    return np.asarray(ret)


def createMaskPhase(phase):
    if phase == 'prep':
        return np.asarray(1)
    if phase.startswith('stroke'):
        return np.asarray(2)
    if phase == 'retr':
        return np.asarray(3)
    if phase == 'post.hold':
        return np.asarray(4)
    if phase == 'pre.hold':
        return np.asarray(5)
    if phase is None:
        return np.asarray(0)
    print(phase)
    return np.asarray(0)


def sg_filter(x, m, k=0):
    """
    x = Vector of sample times
    m = Order of the smoothing polynomial
    k = Which derivative
    """
    mid = len(x) // 2
    a = x - x[mid]
    expa = lambda x: map(lambda i: i**x, a)
    A = np.r_[map(expa, range(0,m+1))].transpose()
    Ai = np.linalg.pinv(A)

    return Ai[k]

def smooth(x, y, order=2, deriv=0):
    if deriv > order:
        raise "deriv must be <= order"

    #n = len(x)
    #m = size

    #result = np.zeros(n)

    f = savgol_filter(y,window_length=5,polyorder=min(4,order),deriv=deriv)
    result = np.dot(f, x)

    #if deriv > 1:
    #    result *= math.factorial(deriv)

    return result


def calculateJerk(d3_array):
    """
        calculate the jerk from the positional data (also calculates velocity and acceleration if needed somewhere)

        :param d3_array: array of gesture data for one hand. (-1,24,3)
    """
    # we could do it with numpy or sklearn, but pandas works.
    # only look at the wrists
    wrist = d3_array[:, 2, :]  # (-1,3)
    wrist = wrist.copy()
    window = 5
    t = np.linspace(0, wrist.shape[0] * (1 / 25), num=wrist.shape[0], dtype=np.float64)
    out = []
    jout = np.zeros((wrist.shape[0], 3))
    for j in range(wrist.shape[0]):

        for i in range(3):

            w1 = wrist[:, i].flatten()
            start = max(0,j - window)
            stop = min(wrist.shape[0], j + window)

            t2 = t[start:stop]
            w2 = w1[start:stop]
            velocity = smooth(t2, w2, order=3, deriv=1).flatten()
            acceleration = smooth(t2, w2, order=3, deriv=2).flatten()
            jerk = smooth(t2, w2, order=3, deriv=3).flatten()
            #print("velocity:",np.sum(np.abs(velocity.flatten())))
            #print("acceleration:", np.sum(np.abs(acceleration.flatten())))
            #print("jerk:", np.sum(np.abs(jerk.flatten())))
            jout[j,i] = jerk
    return jout


def getTargetPhase(leftPhases, rightPhases, cur_T, time_step, length):
    time = cur_T
    ret1 = []
    ret2 = []
    for i in range(length):
        time = time + time_step
        cur_L_Phase, cur_R_Phase = getPhase(leftPhases, rightPhases, time)
        ret1.append(createMaskPhase(cur_L_Phase[2]) if cur_L_Phase is not None else np.asarray(0))
        ret2.append(createMaskPhase(cur_R_Phase[2]) if cur_R_Phase is not None else np.asarray(0))
    return np.asarray(ret1), np.asarray(ret2)


def insertTimings(FS, timings):
    out = []
    for f in FS:
        a, b, c, d = f
        a = timings[a]
        b = timings[b]
        out.append((a, b, c))
    return out


def insertATimings(FS, timings):
    out = []
    for f in FS:
        a, b, c, d = f
        a = timings[a]
        out.append((a[0], a[1], b))
    return out


def removeEmpty(FS):
    out = []
    for f in FS:
        a, b, c, d = f
        if c != "":
            out.append((a, b, c))
    return out



