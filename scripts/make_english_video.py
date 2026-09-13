"""Create English publication videos from saved scores and replay arrays only.

Usage: python scripts/make_english_video.py [--film | --card] [--preview]
The optional 44-second synthetic film requires the previously verified local
replay_*.npz files. The 15-second Nexar film uses docs/figures/x_card.png.
No model is imported, scored, trained, or tuned by this presentation script.
Videos are local exports, excluded from the public Git repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "outputs/synthetic/local_reflex_video"
OUT = ROOT / "outputs/synthetic/local_reflex_video_en"
EXPORT = ROOT / "exports"
W, H, FPS = 1080, 1350, 30
BG, PANEL, LINE = (13, 20, 25), (22, 32, 38), (51, 68, 75)
INK, MUTED = (242, 242, 226), (160, 178, 181)
TEAL, AMBER = (100, 236, 192), (255, 184, 109)
MAX_RESPONSE = .0021
SCENES = [(0, 4, "QUESTION"), (4, 11, "APPROACH"),
          (11, 19, "FAILURE"), (19, 29, "BLIND SPOT"),
          (29, 37, "TEST"), (37, 44, "NEXT")]
DATA = {}
FROZEN = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_replays():
    """Read the original verified replay, never regenerate model responses."""
    report_path = REPLAY / "replay_verification.json"
    report = json.loads(report_path.read_text())
    files = [report_path]
    for key, check in report["selected_cases"].items():
        path = REPLAY / f"replay_{key}.npz"
        files.append(path)
        with np.load(path) as array:
            DATA[key] = {"inputs": array["inputs"], "maps": array["maps"],
                         "signal": array["S"], "ts": array["t_s"], **check}
        item = DATA[key]
        assert item["inputs"].shape == (61, 128, 128)
        assert item["maps"].shape == (61, 5, 5)
        np.testing.assert_array_equal(item["maps"].max(axis=(1, 2)), item["signal"])
        np.testing.assert_allclose(item["signal"][item["ts"] >= .25].max(),
                                   check["peak"], rtol=1e-12, atol=1e-15)
        assert item["maps"].max() <= MAX_RESPONSE
        assert check["max_trace_absolute_error"] == 0
    assert not DATA["bright"]["maps"].any()
    assert DATA["central_bright"]["peak"] > DATA["central_bright"]["theta"]
    assert report["stats"] == {"conditions": 354, "models": 9, "calibration": 18,
        "challenge": 336, "expansion_detected": 84, "expansion_total": 100,
        "nuisance_false_response": 61, "nuisance_total": 236}
    # Verify source files relevant to every displayed quantitative claim.
    for relative, expected in report["source_artifacts_sha256"].items():
        if relative.endswith(("traces.parquet", "case_scores.csv", "cases.json", "config.yaml")):
            path = ROOT / relative
            assert sha(path) == expected, relative
            files.append(path)
    FROZEN.update({str(path.relative_to(ROOT)): sha(path) for path in files})
    (OUT / "replay_provenance.json").write_text(json.dumps({
        "inputs_sha256": FROZEN, "selected_cases": report["selected_cases"],
        "stats": report["stats"], "source_model": "proxy_not_zhao_star",
        "method": "Read saved replay arrays; no model imports or scoring.",
        "display": "Actual 30 Hz samples at 0.5x; final state held. Fixed linear map scale 0..0.0021 AU.",
        "scope": "Separate synthetic experiment, not Nexar dashcam performance."
    }, indent=2) + "\n")


@lru_cache(maxsize=64)
def font(size, weight="regular"):
    if weight == "mono":
        mac = Path("/System/Library/Fonts/Menlo.ttc")
        fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
        return ImageFont.truetype(str(mac if mac.exists() else fallback), size)
    mac = Path("/System/Library/Fonts/HelveticaNeue.ttc")
    fallback = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight == "bold"
                    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(mac if mac.exists() else fallback), size,
                              index=1 if mac.exists() and weight == "bold" else 0)


def txt(im, xy, text, size=30, color=INK, weight="regular", anchor="lt"):
    assert not any("\uac00" <= c <= "\ud7a3" or "\u1100" <= c <= "\u11ff" for c in text)
    draw = ImageDraw.Draw(im)
    bounds = draw.textbbox(xy, text, font=font(size, weight), anchor=anchor)
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > im.width or bounds[3] > im.height:
        raise ValueError(f"Clipped text: {text!r}, {bounds}")
    draw.text(xy, text, font=font(size, weight), fill=color, anchor=anchor)


def rule(im, y, x1=64, x2=1016):
    ImageDraw.Draw(im).line((x1, y, x2, y), fill=LINE, width=1)


def pill(im, xy, text, color=TEAL, size=21):
    x, y = xy
    width = ImageDraw.Draw(im).textlength(text, font=font(size, "bold"))
    ImageDraw.Draw(im).rounded_rectangle((x, y, x+width+28, y+size+18), radius=8,
        fill=tuple(int(c*.12+b*.88) for c, b in zip(color, BG)))
    txt(im, (x+14, y+9), text, size, color, "bold")


def base(scene, t):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    txt(im, (64, 58), "FLY / LOCAL REFLEX", 23, TEAL, "mono")
    txt(im, (1016, 60), "SYNTHETIC RESEARCH FILM", 20, MUTED, "mono", "rt")
    rule(im, 111)
    rule(im, 1216)
    for j, (start, end, label) in enumerate(SCENES):
        x, width = 64+j*162, 142
        d.rounded_rectangle((x, 1243, x+width, 1247), radius=2, fill=LINE)
        frac = float(np.clip((t-start)/(end-start), 0, 1))
        if frac:
            d.rounded_rectangle((x, 1243, x+width*frac, 1247), radius=2, fill=TEAL)
        txt(im, (x, 1270), f"{j+1:02d} {label}", 17, INK if j == scene else MUTED)
    return im


def sample(elapsed, delay=.5):
    return min(int(math.floor(max(0., elapsed-delay)*15+1e-7)), 60)


def input_panel(im, key, idx, x, y, size):
    gray = np.rint(DATA[key]["inputs"][idx]*255).clip(0, 255).astype("uint8")
    panel = Image.fromarray(gray).convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    im.paste(panel, (x, y))
    d = ImageDraw.Draw(im)
    d.rectangle((x-1, y-1, x+size, y+size), outline=LINE, width=1)
    for xx, yy, sx, sy in [(x,y,1,1), (x+size,y,-1,1), (x,y+size,1,-1), (x+size,y+size,-1,-1)]:
        d.line((xx,yy,xx+sx*12,yy), fill=INK, width=2)
        d.line((xx,yy,xx,yy+sy*12), fill=INK, width=2)


def map_panel(im, key, idx, x, y, size):
    d = ImageDraw.Draw(im)
    low, high = np.array(PANEL), np.array(TEAL)
    for row in range(5):
        for col in range(5):
            amplitude = DATA[key]["maps"][idx, row, col]/MAX_RESPONSE
            color = tuple(np.rint(low+amplitude*(high-low)).astype(int))
            d.rounded_rectangle((x+round(col*size/5), y+round(row*size/5),
                x+round((col+1)*size/5)-5, y+round((row+1)*size/5)-5),
                radius=7, fill=color, outline=LINE, width=1)


def arrow(im, x, y, color=MUTED):
    d = ImageDraw.Draw(im)
    d.line((x-23,y,x+23,y), fill=color, width=2)
    d.line((x+15,y-8,x+23,y,x+15,y+8), fill=color, width=2)


def scale(im, y, x=616, width=395):
    d = ImageDraw.Draw(im)
    for j in range(width):
        c = tuple(np.rint(np.array(PANEL)+j/(width-1)*(np.array(TEAL)-np.array(PANEL))).astype(int))
        d.line((x+j,y,x+j,y+7), fill=c)
    txt(im, (x,y+21), "0", 18, MUTED, "mono")
    txt(im, (x+width,y+21), "2.1 × 10⁻³ AU", 19, MUTED, anchor="rt")


def replay_meta(im, key, idx, still=False):
    txt(im, (1016,153), "SHARED LINEAR SCALE · 0–2.1 × 10⁻³ AU", 18, MUTED, anchor="rt")
    t = DATA[key]["ts"][idx]
    state = "matched-time stills" if still else ("final frame held" if idx == 60 else "0.5× replay")
    txt(im, (1016,187), f"Stimulus {t:0.2f} s · {state}", 20, MUTED, anchor="rt")


def trace(im, key, idx, rect, color=TEAL):
    x, y, width, height = rect
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((x,y,x+width,y+height), radius=10, fill=PANEL)
    left, right, top, bottom = x+22, x+width-22, y+18, y+height-22
    d.line((left,bottom,right,bottom), fill=LINE)
    d.rectangle((left,top,left+(right-left)*.125,bottom), fill=(28,40,46))
    pts = [(left+j/60*(right-left), bottom-float(DATA[key]["signal"][j]/MAX_RESPONSE)*(bottom-top))
           for j in range(idx+1)]
    if len(pts) > 1:
        d.line(pts, fill=color, width=3)
    xx, yy = pts[-1]
    d.ellipse((xx-4,yy-4,xx+4,yy+4), fill=color)


def pair_row(im, key, idx, y, label, note, color=TEAL):
    txt(im, (64,y), label, 31, color, "bold")
    txt(im, (1016,y+4), note, 22, MUTED, anchor="rt")
    input_panel(im,key,idx,64,y+54,260)
    arrow(im,383,y+184)
    map_panel(im,key,idx,446,y+54,260)
    txt(im, (762,y+80), "RESPONSE S", 19, MUTED)
    txt(im, (762,y+115), f'{DATA[key]["signal"][idx]*1000:.3f}', 42, color, "mono")
    txt(im, (762,y+169), "× 10⁻³ AU", 22, MUTED)
    trace(im,key,idx,(755,y+219,261,95),color)


def render_scene(scene, elapsed, t):
    im = base(scene,t)
    d = ImageDraw.Draw(im)
    if scene == 0:
        pill(im,(64,156),"INSPIRED BY FLY VISION")
        txt(im,(64,245),"Can a small circuit",76,INK,"bold")
        txt(im,(64,338),"sense an approach?",76,INK,"bold")
        txt(im,(64,466),"A frozen engineering proxy. Tested, including its failures.",31,MUTED)
        idx = sample(elapsed,0)
        replay_meta(im,"loom",idx)
        input_panel(im,"loom",idx,140,598,340)
        arrow(im,546,768,TEAL)
        map_panel(im,"loom",idx,616,598,340)
        txt(im,(140,960),"Synthetic input",25,MUTED)
        txt(im,(616,960),"Recorded model response",25,TEAL)
        txt(im,(64,1081),"A separate synthetic experiment",32,INK)
        txt(im,(64,1132),"Saved model states · 0.5× replay · no connectome simulation",25,MUTED)
    elif scene == 1:
        pill(im,(64,155),"01 / APPROACH")
        txt(im,(64,240),"An expanding disk",72,INK,"bold")
        txt(im,(64,329),"elicits a response.",72,INK,"bold")
        txt(im,(64,435),"SYNTHETIC INPUT · 128 × 128",22,MUTED)
        txt(im,(616,435),"LOCAL RESPONSE · 5 × 5",22,TEAL)
        idx=sample(elapsed)
        replay_meta(im,"loom",idx)
        input_panel(im,"loom",idx,64,486,400)
        arrow(im,541,685,TEAL)
        map_panel(im,"loom",idx,616,486,400)
        txt(im,(64,929),f'Stimulus time {DATA["loom"]["ts"][idx]:.2f} s',23,MUTED)
        scale(im,906)
        txt(im,(64,993),"Maximum local response S",25,MUTED)
        txt(im,(1016,990),f'{DATA["loom"]["signal"][idx]*1000:.3f} × 10⁻³ AU',31,TEAL,anchor="rt")
        trace(im,"loom",idx,(64,1039,952,108))
        txt(im,(64,1168),"All maps and traces share one linear amplitude scale.",21,MUTED)
    elif scene == 2:
        pill(im,(64,155),"02 / COUNTEREXAMPLE",AMBER)
        txt(im,(64,236),"Motion is not always",65,INK,"bold")
        txt(im,(64,316),"an approach.",65,INK,"bold")
        txt(im,(64,414),"INPUT",22,MUTED)
        txt(im,(446,414),"MODEL RESPONSE",22,MUTED)
        idx=sample(elapsed,1)
        replay_meta(im,"fast",idx)
        pair_row(im,"recede",idx,464,"Receding disk","Near zero in this example")
        pair_row(im,"fast",idx,844,"Fast translation","Fixed size · 4× reference speed",AMBER)
        txt(im,(64,1190),"False response: the fast-moving disk does not expand.",21,AMBER)
    elif scene == 3:
        pill(im,(64,155),"03 / THE BLIND SPOT",AMBER)
        txt(im,(64,235),"Same expansion.",66,INK,"bold")
        txt(im,(64,317),"Different brightness.",66,INK,"bold")
        note = ("Totals: 4 off-center positions × 4 sampling schedules" if elapsed >= 6
                else "Matched disks at the same position, left of center.")
        txt(im,(64,418),note,27,MUTED)
        idx=sample(elapsed,1)
        replay_meta(im,"dark",idx)
        pair_row(im,"dark",idx,490,"Dark disk","Off-center: 16/16 detected")
        pair_row(im,"bright",idx,867,"Bright disk","Off-center: 0/16 detected",AMBER)
        if elapsed >= 6:
            txt(im,(64,1187),"A centered bright disk is detected. Position × polarity matters.",24,MUTED)
    elif scene == 4:
        pill(im,(64,155),"04 / MEASURE EVERYTHING")
        txt(im,(64,242),"A few good examples",69,INK,"bold")
        txt(im,(64,328),"are not enough.",69,INK,"bold")
        txt(im,(64,449),"354 synthetic conditions × 9 models",40,TEAL,"bold")
        txt(im,(64,508),"18 calibration / 336 challenge conditions · this proxy below",25,MUTED)
        d.rounded_rectangle((64,592,1016,803),radius=20,fill=PANEL)
        txt(im,(98,625),"Expansion detected",29,INK,"bold")
        txt(im,(979,615),"84 / 100",72,TEAL,"bold","rt")
        txt(im,(98,687),"84%",25,TEAL,"mono")
        for j in range(100):
            x=98+j*8.85
            d.rounded_rectangle((x,750,x+5.5,770),radius=2,fill=TEAL if j<84 else LINE)
        d.rounded_rectangle((64,828,1016,1039),radius=20,fill=PANEL)
        txt(im,(98,861),"Nuisance responses",29,INK,"bold")
        txt(im,(979,849),"61 / 236",72,AMBER,"bold","rt")
        txt(im,(98,920),"25.8%",25,AMBER,"mono")
        for j in range(236):
            x=98+j*3.75
            d.rectangle((x,986,x+2,1006),fill=AMBER if j<61 else LINE)
        txt(im,(64,1092),"The failures are part of the result.",32,INK)
        txt(im,(64,1145),"Synthetic selectivity is not real-road crash performance.",27,MUTED)
    else:
        pill(im,(64,155),"05 / NEXT QUESTION")
        txt(im,(64,245),"Failure defines",69,INK,"bold")
        txt(im,(64,331),"the next experiment.",69,INK,"bold")
        idx=int(np.argmax(DATA["dark"]["signal"]))
        replay_meta(im,"dark",idx,still=True)
        for key,x,label in [("dark",140,"Dark disk"),("bright",640,"Bright disk")]:
            map_panel(im,key,idx,x,488,290)
            txt(im,(x,801),label,25,MUTED)
        txt(im,(537,605),"≠",70,AMBER,"bold","mt")
        txt(im,(64,894),"Why do position and polarity matter?",45,TEAL,"bold")
        txt(im,(64,965),"Tracing the limits of a small visual module.",31,MUTED)
        rule(im,1041)
        txt(im,(64,1072),"Frozen engineering proxy · synthetic inputs · no training",25,INK)
        txt(im,(64,1123),"proxy_not_zhao_star",25,MUTED,"mono")
        txt(im,(64,1164),"github.com/spacegiyou/fly-ttc",24,TEAL)
    return im


def film_frame(t):
    scene=max(j for j,(start,_,_) in enumerate(SCENES) if t>=start)
    start=SCENES[scene][0]
    current=render_scene(scene,t-start,t)
    if scene and t-start<.2:
        previous=render_scene(scene-1,SCENES[scene-1][1]-SCENES[scene-1][0]-.001,start-.001)
        frac=(t-start)/.2
        current=Image.blend(previous,current,frac*frac*(3-2*frac))
    return current


@lru_cache(maxsize=1)
def card_image():
    path=ROOT/"docs/figures/x_card.png"
    image=Image.open(path).convert("RGB")
    FROZEN[str(path.relative_to(ROOT))]=sha(path)
    return image


def card_frame(t):
    im=Image.new("RGB",(1080,1080),BG)
    d=ImageDraw.Draw(im)
    txt(im,(64,59),"FLY-TTC",25,TEAL,"mono")
    txt(im,(1016,60),"A NEGATIVE RESULT",22,MUTED,"mono","rt")
    rule(im,111)
    if t<3:
        pill(im,(64,163),"PUBLIC NEXAR DASHCAM · FROZEN PROXY")
        txt(im,(64,290),"Not a 166k-neuron",78,INK,"bold")
        txt(im,(64,387),"drive demo",78,INK,"bold")
        txt(im,(64,556),"A looming proxy,",48,TEAL,"bold")
        txt(im,(64,620),"tested on 200 real clips.",48,TEAL,"bold")
        txt(im,(64,778),"100 clips, then a disjoint 100.",32,MUTED)
        txt(im,(64,831),"No connectome simulation. No training.",32,MUTED)
    elif t<8:
        # Keep the source chart unchanged, at its full readable aspect ratio.
        card=card_image()
        ratio=min(1080/card.width,900/card.height)
        resized=card.resize((round(card.width*ratio),round(card.height*ratio)),Image.Resampling.LANCZOS)
        im.paste(resized,((1080-resized.width)//2,130+(900-resized.height)//2))
    elif t<12:
        pill(im,(64,163),"DISJOINT CONFIRMATION · 100 CLIPS")
        txt(im,(64,279),"Confirmation AUROC",62,INK,"bold")
        txt(im,(54,386),"0.49",210,AMBER,"bold")
        txt(im,(64,657),"At chance on this sample.",42,INK,"bold")
        txt(im,(64,770),"Camera-motion subtraction did not help.",32,MUTED)
        txt(im,(64,828),"Even frame difference scored 0.53.",32,MUTED)
    else:
        pill(im,(64,163),"CODE · SCORES · REPRODUCIBLE FIGURES")
        txt(im,(64,286),"Global expansion",74,INK,"bold")
        txt(im,(64,379),"is not collision.",74,INK,"bold")
        txt(im,(64,566),"github.com/spacegiyou/",49,TEAL,"bold")
        txt(im,(64,632),"fly-ttc",80,TEAL,"bold")
        txt(im,(64,830),"Frozen proxy. Public data. An honest negative result.",29,MUTED)
    if not 3<=t<8:
        rule(im,966)
        txt(im,(64,1001),"NEXAR · NOT A CONNECTOME",22,MUTED,"mono")
    d.rectangle((0,1074,1080*t/15,1080),fill=TEAL)
    return im


def preview(frame_func,times,prefix):
    thumbw,thumbh=270,round(frame_func(times[0]).height/4)
    cols=4
    rows=math.ceil(len(times)/cols)
    sheet=Image.new("RGB",(cols*thumbw,rows*(thumbh+26)),(6,10,13))
    d=ImageDraw.Draw(sheet)
    for j,t in enumerate(times):
        frame=frame_func(t)
        frame.save(OUT/f"{prefix}_{t:04.1f}s.png")
        x,y=(j%cols)*thumbw,(j//cols)*(thumbh+26)
        sheet.paste(frame.resize((thumbw,thumbh),Image.Resampling.LANCZOS),(x,y))
        d.text((x+8,y+thumbh+4),f"{t:.1f}s",font=font(15,"mono"),fill=INK)
    sheet.save(OUT/f"{prefix}_contact_sheet.png")


def encode(frame_func,duration,size,path):
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    command=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo",
             "-vcodec","rawvideo","-pix_fmt","rgb24","-s",f"{size[0]}x{size[1]}",
             "-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","medium",
             "-vf","scale=in_range=full:out_range=tv:out_color_matrix=bt709",
             "-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",
             "-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709",str(path)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
        for j in range(FPS*duration):
            proc.stdin.write(frame_func(j/FPS).tobytes())
            if j%(FPS*4)==0:
                print(f"{path.name}: {j/FPS:.0f}/{duration}s",flush=True)
        proc.stdin.close()
        if proc.wait()!=0:
            raise RuntimeError("Video encode failed")
    info=json.loads(subprocess.check_output([shutil.which("ffprobe"),"-v","error",
        "-show_format","-show_streams","-of","json",str(path)]))
    video=next(s for s in info["streams"] if s["codec_type"]=="video")
    assert video["codec_name"]=="h264" and video["pix_fmt"]=="yuv420p"
    assert (video["width"],video["height"])==size
    assert int(video["nb_frames"])==FPS*duration
    assert abs(float(info["format"]["duration"])-duration)<.1
    assert path.stat().st_size<20_000_000
    # A full decode catches corruption, in addition to metadata checks.
    subprocess.run([ffmpeg,"-hide_banner","-v","error","-i",str(path),"-f","null","-"],check=True)
    info["sha256"]=sha(path)
    (OUT/f"{path.stem}_probe.json").write_text(json.dumps(info,indent=2)+"\n")
    print(f"Verified {path}: {path.stat().st_size:,} bytes",flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--preview",action="store_true")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--film",action="store_true")
    group.add_argument("--card",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    EXPORT.mkdir(exist_ok=True)
    if not args.card:
        prepare_replays()
        preview(film_frame,[.7,3.5,6.8,10.,13.6,15.7,18.,21.7,23.7,27.,33.,40.],"film_en")
        if not args.preview:
            encode(film_frame,44,(W,H),EXPORT/"fly_ttc_reflex_film_en.mp4")
    if not args.film:
        card_image()
        preview(card_frame,[1.,5.,10.,13.],"nexar_en")
        if not args.preview:
            encode(card_frame,15,(1080,1080),EXPORT/"fly_ttc_nexar_card_en.mp4")
    for relative,expected in FROZEN.items():
        assert sha(ROOT/relative)==expected,f"Source changed during video production: {relative}"
    print("Saved research inputs unchanged. All burned-in captions are English.",flush=True)


if __name__=="__main__":
    main()
