# Press the green button in the gutter to run the script.
import codecs
import os
import re
import traceback
from collections import Counter

import pympi
import torch
import yaml
from torch import package

from helper_function import insertTimings


def apply_te(text, lan='de'):
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                   'latest_silero_models.yml',
                                   progress=False)

    with codecs.open('latest_silero_models.yml', 'r', "utf-8") as yaml_file:
        models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    model_conf = models.get('te_models').get('latest')
    # see avaiable languages
    available_languages = list(model_conf.get('languages'))
    print(f'Available languages {available_languages}')

    # and punctuation marks
    available_punct = list(model_conf.get('punct'))
    print(f'Available punctuation marks {available_punct}')

    model_url = model_conf.get('package')

    model_dir = "downloaded_model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))

    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url,
                                       model_path,
                                       progress=True)

    imp = package.PackageImporter(model_path)
    model = imp.load_pickle("te_model", "model")
    return model.enhance_text(text, lan)


def buildSentence(FS, stopTime, time_threshold):
    FS = FS.copy()
    triggered = False
    out = []
    cur_sen = ""
    start_t = FS[0][0]
    cur_t = FS[0][0]
    while not triggered and len(FS) > 0:
        fs = FS[0]
        if fs[0] > stopTime or fs[2] == "":
            if fs[2] == "":
                FS.pop(0)
            triggered = True
        else:
            FS.pop(0)
            if fs[0] > cur_t + time_threshold:# or fs[2] == '<SILENCE>':
                out.append((start_t, cur_t, cur_sen.strip()))
                cur_sen = fs[2]
                start_t = fs[0]
            else:
                cur_sen = cur_sen + " " + fs[2]
            cur_t = fs[1]
    if cur_sen.strip() != "":
        out.append((start_t, cur_t, cur_sen.strip()))

    return out, FS

def removeEmpty(S):
    ret = []
    for FSS in S:
        if FSS[2] != '':
            ret.append(FSS)
    return ret

def createDialogue_new(file_path,FS,RS, timings,time_threshold):
    FS = list(FS[0].values())
    RS = list(RS[0].values())
    FS = insertTimings(FS, timings)
    RS = insertTimings(RS, timings)

    FS = removeEmpty(FS)
    RS = removeEmpty(RS)
    ct = 0
    current_sentence = ""
    L1 = "FS"
    L2 = "RS"

    if RS[0][0] < FS[0][0]:
        L1 = "RS"
        L2 = "FS"
        FS1 = FS
        FS = RS
        RS = FS1
    ret = []
    state = 0
    start_t = 0
    while len(FS) > 0 and len(RS) > 0:
        if state == 0 and len(FS) > 0:
            st,et,w = FS[0]
            FS.remove(FS[0])
            current_sentence += w + " "
            ct = et
            if (len(RS) > 0 and RS[0][0] <= ct) or len(current_sentence) > 400:
                state = 1

                ret.append((L1,start_t,et,current_sentence.strip()))
                start_t = RS[0][0]
                current_sentence = ""
        else:
            if len(RS) > 0:
                st, et, w = RS[0]
                RS.remove(RS[0])
                current_sentence += w + " "
                ct = et
                if (len(FS) > 0 and FS[0][0] <= ct) or len(current_sentence) > 400:
                    state = 0
                    ret.append((L2,start_t,et,current_sentence.strip()))
                    start_t = FS[0][0]
                    current_sentence = ""
            else:
                state = 0
    with codecs.open(file_path + ".txt", "w", "utf-8") as f:
        for sp,st,et,sen in ret:
            if not sen.endswith(".") and not sen.endswith("?") and not sen.endswith("!"):
                sen = sen + "."
            f.write(str(sp) + "; " + str(st) + "; " + str(et) + "; " + sen + "\n")
    return ret

def createMonologe_new(file_path,FS,RS, timings,time_threshold):
    FS = list(FS[0].values())
    RS = list(RS[0].values())
    FS = insertTimings(FS, timings)
    RS = insertTimings(RS, timings)

    FS = removeEmpty(FS)
    RS = removeEmpty(RS)

    mono = RS if len(RS) > len(FS) else FS

    ret = []
    state = 0
    start_t = 0
    current_sentence = ""
    while len(mono) > 0:
            st,et,w = mono[0]
            mono.remove(mono[0])
            current_sentence += w + " "
            ct = et
            if len(mono) == 0 or len(current_sentence) > 100:
                ret.append(("RS",start_t,et,current_sentence.strip()))
                start_t = et
                current_sentence = ""

    with codecs.open(file_path + ".txt", "w", "utf-8") as f:
        for sp,st,et,sen in ret:
            if not sen.endswith(".") and not sen.endswith("?") and not sen.endswith("!"):
                sen = sen + "."
            f.write(str(sp) + "; " + str(st) + "; " + str(et) + "; " + sen + "\n")
    return ret







def createDialogue(file_path, FS, RS, timings, time_threshold=2000):
    FS = list(FS[0].values())
    RS = list(RS[0].values())
    FL = "FS"
    RL = "RS"
    FS = insertTimings(FS, timings)
    RS = insertTimings(RS, timings)

    dialog = []
    while len(FS) > 0 and len(RS) > 0:
        d1, FS = buildSentence(FS, RS[0][0], time_threshold)
        for d in d1:
            dialog.append([FL, d[0], d[1], d[2]])
        d2, RS = buildSentence(RS, FS[0][0] if len(FS) > 0 else 10000000000000000, time_threshold)
        for d in d2:
            dialog.append([RL, d[0], d[1], d[2]])
    print(dialog)
    checkL = []
    see = [("son", "so ein"), ("sone", "so eine"), ("ne", "eine"), ("rum", "herum"), ("gings", "ging es"),
           ("sonem", "so einem")]
    with codecs.open(file_path + ".txt", "w", "utf-8") as f:
        for item in dialog:
            time = item[0]
            dial = item[1]
            text = str(item[3]).lower().replace(" son ", " so ein ").replace(" sone ", " so eine ").replace(" ne ",
                                                                                                            " eine ").replace(
                " rum ", " herum ").replace(" gings ", " ging es ")
            for s in see:
                text = re.sub(r"\b%s\b" % s[0], s[1], text)
            comb_L = checkL[-5:]
            apply_t = " ".join(comb_L)
            splitSTR = apply_t[-5:]
            checkL.append(text)
            apply_t += " " + text

            try:
                texter = apply_te(apply_t)
                if splitSTR != "":
                    text_1 = texter.split(splitSTR)
                    if len(text_1) == 2:
                        text = text_1[-1]
                    else:
                        text = apply_te(text)
            except Exception:
                traceback.print_exc()
            if not text.endswith(".") and not text.endswith("!") and not text.endswith("?"):
                text += "."
            if text.startswith(".") or text.startswith("?") or text.startswith("!") or text.startswith(","):
                text = text[1:]

            text = text.strip()
            if text.endswith(" nee.") or text.endswith(" nee?") or text.endswith(" nee!"):
                text = text.replace(" nee.", ", oder?")

            print(text)
            if len(text) > 1:
                f.write(str(item[0]) + "; " + str(item[1]) + "; " + str(item[2]) + "; " + text + "\n")
        f.flush()


def getDI(ee):
    timings = ee.timeslots
    RS1 = list(ee.tiers["R.G.Left.Phrase"][0].values())
    RS2 = list(ee.tiers["R.G.Right.Phrase"][0].values())

    FS1 = list(ee.tiers["F.G.Left.Phrase"][0].values())
    FS2 = list(ee.tiers["F.G.Right.Phrase"][0].values())

    FS1 = insertTimings(FS1, timings)
    FS2 = insertTimings(FS2, timings)

    RS1 = insertTimings(RS1, timings)
    RS2 = insertTimings(RS2, timings)

    RS = RS1 + RS2
    FS = FS1 + FS2

    return FS, RS


def getWordsFromSemantic(RS, RS2):
    out = []
    for SE in RS2:
        wout = ""
        start = SE[0]
        end = SE[1]
        type = SE[2]
        for ww in RS:
            if ww[0] >= start - 500 and ww[1] <= end + 500:
                wout += ww[2] + " "
        out.append((wout, start - 500, end + 500, type))
    return out


def getReference(ee):
    timings = ee.timeslots
    RS = list(ee.tiers["R.S.Form"][0].values())
    RS2 = list(ee.tiers["R.S.Semantic Feature" if "R.S.Semantic Feature" in ee.tiers.keys() else "R.S.Sematic Feature"][
                   0].values())
    RS = insertTimings(RS, timings)
    RS2 = insertTimings(RS2, timings)

    words_in = getWordsFromSemantic(RS, RS2)
    words_out = ReferencesFromNL(words_in)
    return words_out


def ReferencesFromNL(words_in):
    out = []
    for win in words_in:
        wine = win[0]
        start = win[1]
        end = win[2]
        type = win[3]

        words = wine.split()
        main = []
        for w in words:
            if w[0].isupper():
                main.append(w)
        numbers = -1
        # print(type)
        if "Amount" in type:
            number_one = ["ein", "son", "eine", "eins", "ein", "einem", "dieser", "ersten", "einen", "eingeschossig",
                          "jedem", "jeweils"]
            number_two = ["beiden", "beide", "zwei", "zweigeschossig", "doppelt", "doppelter", "zwillingsturm",
                          "zweiten"]
            number_three = ["drei", "dreigeschossig", "dritten"]
            number_more = ["vier", "sechs", "acht", "acht-", "zwanzig", "zehn", "fünfzehn", "fünf", "zehn", "zwanzig"]
            number = "ein"
            for w in words:
                if w.lower() in number_three:
                    numbers = "drei"
                if w.lower() in number_two:
                    numbers = "zwei"
                if w.lower() in number_one:
                    numbers = "ein"
                if w.lower() in number_more:
                    numbers = "viele"
        position = -1
        if "relative Position" in type:
            position_infront = ["vor", "vorderseiten", "vordach", "anfang", "erste", "front", "davor", "vorm", "vorne",
                                "hin", "vorderseite", "vorbau", "frontbau"]
            position_behind = ["dahinter", "hinter", "hinterm", "durch", "hinteren", "rum", "dahinten", "weiter"]
            position_left = ["linken", "links", "linke"]
            position_right = ["rechten", "rechts", "rechte", "recht", "recht..."]
            position_both_directions = ["links und rechts", "Seiten"]
            position_atop = ["auf", "drauf", "übereinander", "oben", "zweite", "über", "aufm", "spitzen", "spitze",
                             "aufgesetzten", "hochgehoben", "drüber", "übereinenander", "übereienander", "obere",
                             "darauf", "bereinander", "dritten", "zweiten"]
            position_below = ["unten", "untere"]
            position_around = ["um", "drum", "nebeneinander", "zusammen", "im", "seite", "mitte", "umgeben",
                               "eingefasst", "außen", "umzäunt", "daneben", "am", "eingezäunt", "an", "nebengängen",
                               "verbunden", "quer"]
            position_inside = ["drinnen", "innen", "in", "zwischen", "ineinander", "in", "dazwischen", "drin"]

            for p in position_both_directions:
                if p in wine:
                    position = "links und rechts"
            if position == -1:
                for w in words:
                    if w.lower() in position_infront:
                        position = "davor"
                    if w.lower() in position_behind:
                        position = "hinter"
                    if w.lower() in position_left:
                        position = "links"
                    if w.lower() in position_right:
                        position = "rechts"
                    if w.lower() in position_atop:
                        position = "über"
                    if w.lower() in position_below:
                        position = "drunter"
                    if w.lower() in position_inside:
                        position = "innen"
                    if w.lower() in position_around:
                        position = "aussen"
            if position == -1:
                position = ""
        shape = -1
        if "Shape" in type:
            shape_rund = ["runden", "abgerundeten", "rund", "rundes", "runde", "betonring", "rundfenster", "rosette",
                          "bogentür", "säule", "kreisel", "glaskugel", "rundbogen", "kreisförmig", "ring", "untertasse",
                          "rundbögen", "rundere", "kugelabschnitt", "kreisrunder", "runddach", "kreis", "rundkuppel",
                          "runder", "kugel"]
            shape_schlaengel = ["treppen", "geschlängelte", "geschwungen", "s-kurve", "helixförmig"]
            shape_eckig = ["rechteckiger", "eckige", "eckig", "quadratische", "quader", "würfel", "viereckig",
                           "quadrat", "viereckiger", "eckiges", "klotz", "quadrat", "quadratisch", "zehneckig"]
            shape_U = ["u-förmige", "u-förmiges", "u", "y", "umgedrehte", "hufeisen", "u-förmig", "kelchförmig",
                       "halbmond", "halbkugeln", "halbkugel", "halbkreis", "u-förmiger", "u-ende", "halb", "geöffnet",
                       "us"]
            shape_T = ["t", "t-kreuzung"]
            shape_cross = ["kreuz", "kreuzform", "kreuzaufbau"]
            shape_spitz = ["spitzdach", "spitzen", "spitze", "spitz", "spitzer", "pyramide", "spitzes",
                           "dreiecksförmig", "spitzt"]
            shape_flach = ["flach", "flachbau", "walldach", "walmdach", "flachdach"]
            for w in words:
                if w.lower() in shape_rund:
                    shape = "rund"
                if w.lower() in shape_schlaengel:
                    shape = "schlängelnd"
                if w.lower() in shape_eckig:
                    shape = "eckig"
                if w.lower() in shape_U:
                    shape = "U-Form"
                if w.lower() in shape_T:
                    shape = "T-Form"
                if w.lower() in shape_spitz:
                    shape = "spitz"
                if w.lower() in shape_cross:
                    shape = "kreuz"
                if w.lower() in shape_flach:
                    shape = "flach"
            if shape == -1:
                shape = ""

        Entity = -1
        if "Entity" in type:
            entity_dict = {}
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
                                      "balkon", "balkone", "außengebäuden", "gebaudeteil"]
            entity_dict["Kunstobjekt"] = ["säule", "ding", "form", "betonsockel", "röhren", "roehren", "ring",
                                          "stangen", "betonsäulue", "betonsäule"]
            entity_dict["Dach"] = ["rundkuppel", "spitzdach", "vordach", "kuppel", "flachbau", "dach", "giebel",
                                   "walldach", "giebel", "walmdach", "flachdach", "runddach", "kirchdach", "dächern",
                                   "spitze"]
            entity_dict["Brunnen"] = ["brunnen"]
            entity_dict["Hecke"] = ["hecke"]
            entity_dict["X"] = ["aufgänge", "glasvorbau", "enden", "vorbau", "rohr", "öffnung", "objekt", "einkerbung",
                                "frontbau", "ecken", "pyramide", "höhe", "radius", "außenteile", "glassvorbau",
                                "teilen", "flügeln", "rundbogen", "doppelkaskade", "kugelabschnitt", "einkerbung",
                                "kreuzaufbau", "erhebung", "seite", "Doppelkaskade", "klotz", "portal",
                                "Kugelabschnitt", "einkerbungen", "seiten", "bogen", "kanten", "stück", "rahmen",
                                "anfang", "front", "meter", "riesenteil", "vorderseite", "teil", "bereich", "block",
                                "kreuzform", "ende", "stahlbogen", "metallwand", "mauer", "platten"]

            for w in main:
                for k in entity_dict.keys():
                    if w.lower() in entity_dict[k]:
                        Entity = k
            if Entity == -1:
                Entity = "X"

        size = -1
        if "Size" in type:
            size_dict = {}
            size_dict["groß"] = ["breites", "höchste", "breiteres", "zehn", "zwanzig", "fünfzehn", "groß", "groß",
                                 "größer", "großer", "großes", "großen", "größere", "große", "riesenteil", "größerer",
                                 "riesenplatz", "dreigeschossig", "eingeschossig", "zweigeschossig"]
            size_dict["klein"] = ["klein", "kleine", "kleiner", "kleinem", "kleinere", "kleines", "kleinen", "kurze",
                                  "kurz", "kleinerer", "kurzen", "kleinerer"]
            for k in size_dict.keys():
                size_dict[k] = [t.lower() for t in size_dict[k]]

            for w in words:
                for k in size_dict.keys():
                    if w.lower() in size_dict[k]:
                        size = k
            if size == -1:
                size = ""
        deictic = -1
        if "Deictic" in type:
            deictic_dict = {}
            deictic_dict["hier"] = ["hier"]
            deictic_dict["da"] = ["da"]
            deictic_dict["so"] = ["so", "son", "solche", "sonem", "sowas", "eine"]
            deictic_dict["diese"] = ["diese", "die"]
            for k in deictic_dict.keys():
                deictic_dict[k] = [t.lower() for t in deictic_dict[k]]

            for w in words:
                for k in deictic_dict.keys():
                    if w.lower() in deictic_dict[k]:
                        deictic = k
            if deictic == -1:
                deictic = ""

        out_solved_type = ""  # amount,relative position,shape,entity,size,deictic
        out_solved_type += "Amount: " + (str(numbers) if numbers != -1 else "") + ", "
        out_solved_type += "Position: " + (str(position) if position != -1 else "") + ", "
        out_solved_type += "Shape: " + (str(shape) if shape != -1 else "") + ", "
        out_solved_type += "Size: " + (str(size) if size != -1 else "") + ", "
        out_solved_type += "Deictic: " + (str(deictic) if deictic != -1 else "") + ", "
        out_solved_type += "Entity: " + (str(Entity) if Entity != -1 else "")

        out_clearname_type = ""
        out_clearname_type += (str(deictic) + " " if deictic != -1 else "")
        out_clearname_type += (str(position) + " " if position != -1 else "")
        out_clearname_type += (str(numbers) + " " if numbers != -1 else "")
        out_clearname_type += (str(size) + " " if size != -1 else "")
        out_clearname_type += (str(shape) + " " if shape != -1 else "")
        out_clearname_type += (str(Entity) if Entity != -1 else "")

        out.append((wine, start, end, type, out_solved_type, out_clearname_type,
                    (deictic, position, numbers, size, shape, Entity)))

    return out


if __name__ == '__main__':
    for root, subdirs, files in os.walk("/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/Owncloud/data/video/"):

        # for subdir in subdirs:
        #    print('\t- subdirectory ' + subdir)
        words_out = []
        phases = []
        for filename in files:
            # print(filename)
            file_path = os.path.join(root, filename)
            if file_path.endswith(".eaf"):
                ee = pympi.Elan.Eaf(file_path)
                RS = ee.tiers["R.S.Form"]
                FS = ee.tiers["F.S.Form"] if "F.S.Form" in ee.tiers.keys() else ee.tiers["F.S.Form "]
                timings = ee.timeslots
                createMonologe_new(file_path, FS, RS, timings,time_threshold=100000000)
                # RS = insertTimings(ee.tiers["R.S.Form"][0].values(),ee.timeslots)
                # RS = buildSentence(RS, 100000000000000, 1000000000000000000)
                # print(RS)
                #e1 = getReference(ee)
                #for eout in e1:
                #    words_out.append(str(eout[6][5]).lower())
                #    print(e1)
                # getHandPositions(ee)

                # phases += getPhases(ee)
                # annotationsToTimings(ee)
                # words_out += getReference(ee)
        # cc = []
        # for l1,r1 in phases:
        #     for l2 in l1:
        #         cc.append(l2[2])
        #     for r2 in r1:
        #         cc.append(r2[2])
        print(Counter(words_out))
        # for cc in ccountLeft:
        #     kk = Counter(cc)
        #     sorted_x = reversed(sorted(kk.items(), key=lambda kv: kv[1]))
        #     for so in sorted_x:
        #         if so[1] >= 50:
        #             print(so[0],":",so[1]," ",end="")
        #     print("\n--------------\n")
        # print(Counter([c[5] for c in words_out]))
        # calc = np.zeros((6,6))
        # for c in words_out:
        #     fet = c[6]
        #     for i in range(6): # deictic positon numbers size shape entity
        #         for j in range(6):
        #             if fet[i] != -1:
        #                 if fet[j] != -1:
        #                     calc[i,j] += 1
        #
        #
        # print(calc)
        # print(Counter([c[6][i] for c in words_out]))
        # getDI(ee)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
