import numpy as np

from SkeletonHelper.helper_function import getAngle


def calculateCangle(arr):
    a, b, c = arr
    a = a.copy()
    b = b.copy()
    c = c.copy()
    return 180 - getAngle(a, b, c)


def calculateAngleOneHand(hand_points):
    thumbIP = calculateCangle([hand_points[2], hand_points[3], hand_points[4]])
    thumbMCP = calculateCangle([hand_points[1], hand_points[2], hand_points[3]])

    indexDIP = calculateCangle([hand_points[6], hand_points[7], hand_points[8]])
    indexPIP = calculateCangle([hand_points[5], hand_points[6], hand_points[7]])
    indexMCP = calculateCangle([hand_points[0], hand_points[5], hand_points[6]])

    middleDIP = calculateCangle([hand_points[10], hand_points[11], hand_points[12]])
    middlePIP = calculateCangle([hand_points[9], hand_points[10], hand_points[11]])
    middleMCP = calculateCangle([hand_points[0], hand_points[9], hand_points[10]])

    ringDIP = calculateCangle([hand_points[14], hand_points[15], hand_points[16]])
    ringPIP = calculateCangle([hand_points[13], hand_points[14], hand_points[15]])
    ringMCP = calculateCangle([hand_points[0], hand_points[13], hand_points[14]])

    littleDIP = calculateCangle([hand_points[18], hand_points[19], hand_points[20]])
    littlePIP = calculateCangle([hand_points[17], hand_points[18], hand_points[19]])
    littleMCP = calculateCangle([hand_points[0], hand_points[17], hand_points[18]])

    return [thumbMCP, thumbIP, indexMCP, indexPIP, indexDIP, middleMCP, middlePIP, middleDIP, ringMCP, ringPIP, ringDIP,
            littleMCP, littlePIP, littleDIP]


def calculateHandStiffness(lAngle, rAngle):
    relaxangle = np.asarray([46.1, 9.3, 28.8, 25.5, 14, 33, 30.1, 13.1, 25.6, 36.1, 10.6, 17.1, 33.3, 16.9])
    return lAngle - relaxangle, rAngle - relaxangle


def relaxedByFinger(hand):
    thumb = np.asarray([hand[0], hand[1]]).mean()
    index = np.asarray([hand[2], hand[3], hand[4]]).mean()
    middle = np.asarray([hand[5], hand[6], hand[7]]).mean()
    ring = np.asarray([hand[8], hand[9], hand[10]]).mean()
    little = np.asarray([hand[11], hand[12], hand[13]]).mean()
    return thumb, index, middle, ring, little


def getFingerAnnotationPos(hand_points):
    return [hand_points[2], hand_points[3], hand_points[5], hand_points[5], hand_points[7],
            hand_points[9], hand_points[10], hand_points[11], hand_points[13], hand_points[14], hand_points[15],
            hand_points[17], hand_points[18], hand_points[19]]


def getPointFingerAnnotationPos(hand_points):
    return [hand_points[4], hand_points[8], hand_points[12], hand_points[16], hand_points[20]]


def getHandFingerAnnotationPos(d3_pose):
    lpoints = d3_pose[17:17 + 21]
    rpoints = d3_pose[17 + 21:]
    return getFingerAnnotationPos(lpoints), getFingerAnnotationPos(rpoints)


def getHandFingerTipAnnotationPos(d3_pose):
    lpoints = d3_pose[17:17 + 21]
    rpoints = d3_pose[17 + 21:]
    return getPointFingerAnnotationPos(lpoints), getPointFingerAnnotationPos(rpoints)


def calculateAngleForHands(d3_pose):
    lpoints = d3_pose[17:17 + 21]
    rpoints = d3_pose[17 + 21:]
    return np.asarray(calculateAngleOneHand(lpoints)), np.asarray(calculateAngleOneHand(rpoints))


def nicePrintHandTable(lpoints, rpoints):
    print("-----------------")
    for i, pp in enumerate([lpoints, rpoints]):
        print("Thumb", "Left:" if i == 0 else "Right:")
        print("MCP: {:.2f}".format(pp[0]), "IP: {:.2f}".format(pp[1]))
        print("Index finger", "Left:" if i == 0 else "Right:")
        print("MCP: {:.2f}".format(pp[2]), "PIP: {:.2f}".format(pp[3]), "DIP: {:.2f}".format(pp[4]))
        print("Middle finger", "Left:" if i == 0 else "Right:")
        print("MCP: {:.2f}".format(pp[5]), "PIP: {:.2f}".format(pp[6]), "DIP: {:.2f}".format(pp[7]))
        print("Ring finger", "Left:" if i == 0 else "Right:")
        print("MCP: {:.2f}".format(pp[8]), "PIP: {:.2f}".format(pp[9]), "DIP: {:.2f}".format(pp[10]))
        print("Little finger", "Left:" if i == 0 else "Right:")
        print("MCP: {:.2f}".format(pp[11]), "PIP: {:.2f}".format(pp[12]), "DIP: {:.2f}".format(pp[13]))
    print("-----------------")