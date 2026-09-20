from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


@dataclass(frozen=True)
class ConceptSpec:
    word: str
    domain: str
    wnids: frozenset[str]


@dataclass(frozen=True)
class Placement:
    geometry: str
    row: int
    col: int


@dataclass(frozen=True)
class RenderStyle:
    font_path: str
    font_size: int
    stroke_width: int


@dataclass
class Rendered:
    image: Image.Image
    mask: Image.Image
    coords: List[Tuple[int, int]]
    style: RenderStyle
    ink_bbox: Tuple[int, int, int, int]


@dataclass
class TinyPatchPacketData:
    images: List[Image.Image]
    masks: List[Image.Image]
    mask_weights: List[float]
    present_targets: List[float]
    readable_targets: List[float]
    captions: List[str]
    positive_pairs: List[Tuple[int, int]]
    source_triplets: List[Tuple[int, int, List[int]]]
    auto_triplets: List[Tuple[int, int, int]]
    invariance_pairs: List[Tuple[int, int]]
    metadata: Dict[str, Any]


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little") & 0x7FFFFFFF


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def crop_like_training(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, BICUBIC)
    left = max(0, (resized[0] - size) // 2)
    top = max(0, (resized[1] - size) // 2)
    return image.crop((left, top, left + size, top + size))


def _wnid_num(wnid: str) -> Optional[int]:
    match = re.fullmatch(r"n(\d{8})", str(wnid))
    return int(match.group(1)) if match else None


def _find_wnids_by_exact_label(lookup: Mapping[str, str], labels: Iterable[str]) -> set[str]:
    wanted = {str(x).strip().lower() for x in labels}
    return {wnid for wnid, label in lookup.items() if str(label).strip().lower() in wanted}


def build_short_concept_taxonomy(lookup: Mapping[str, str]) -> Dict[str, ConceptSpec]:
    """Curated <=3-character semantic superlabels over ImageNet-1k classes.

    This deliberately treats ordinary ImageNet dog breeds as DOG and domestic
    cat synsets as CAT. The purpose is controlled semantic crossing, not a full
    WordNet reconstruction.
    """
    dog_wnids = {
        wnid for wnid in lookup
        if (_wnid_num(wnid) is not None and 2085620 <= int(_wnid_num(wnid)) <= 2113978)
    }
    dog_wnids |= {w for w in ("n02115641", "n02115913", "n02116738") if w in lookup}

    exact: Dict[str, Tuple[str, Sequence[str]]] = {
        "cat": ("animal", ("tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat")),
        "fox": ("animal", ("kit fox", "red fox", "Arctic fox", "grey fox")),
        "pig": ("animal", ("hog", "wild boar", "warthog")),
        "ram": ("animal", ("ram",)),
        "hen": ("animal", ("hen",)),
        "bee": ("animal", ("bee",)),
        "ant": ("animal", ("ant",)),
        "fly": ("animal", ("fly",)),
        "owl": ("animal", ("great grey owl",)),
        "eel": ("animal", ("eel",)),
        "gar": ("animal", ("gar",)),
        "jay": ("animal", ("jay",)),
        "ox":  ("animal", ("ox",)),
        "car": ("vehicle", ("passenger car", "sports car", "Model T", "convertible", "limousine", "minivan", "cab", "racer", "jeep", "beach wagon")),
        "bus": ("vehicle", ("school bus", "minibus", "trolleybus")),
        "van": ("vehicle", ("moving van", "police van")),
        "cup": ("object", ("cup", "measuring cup")),
        "ski": ("object", ("ski",)),
        "sax": ("object", ("sax",)),
        "bow": ("object", ("bow",)),
        "dam": ("place", ("dam",)),
        "fig": ("food", ("fig",)),
    }

    out: Dict[str, ConceptSpec] = {"dog": ConceptSpec("dog", "animal", frozenset(dog_wnids))}
    for word, (domain, labels) in exact.items():
        out[word] = ConceptSpec(word, domain, frozenset(_find_wnids_by_exact_label(lookup, labels)))
    return out


def parse_group_wnid(group_id: str, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    match = re.match(r"^(n\d{8})", str(group_id))
    if match:
        return match.group(1)
    for row in rows:
        for key in ("wnid", "class_wnid", "synset"):
            value = row.get(key)
            if value and re.fullmatch(r"n\d{8}", str(value)):
                return str(value)
        for key in ("image_relpath", "image"):
            value = row.get(key)
            if value:
                match = re.search(r"(n\d{8})", str(value))
                if match:
                    return match.group(1)
    return None


def _taxonomy_membership(wnid: str, taxonomy: Mapping[str, ConceptSpec]) -> List[str]:
    return sorted(word for word, spec in taxonomy.items() if wnid in spec.wnids)


def occupied_patch_coords(mask: Image.Image, patch_size: int, threshold: int = 8) -> List[Tuple[int, int]]:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0:
        return []
    return sorted({(int(y // patch_size), int(x // patch_size)) for y, x in zip(ys, xs)})


def _local_luminance(image: Image.Image, box: Tuple[int, int, int, int]) -> float:
    arr = np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.5
    return float((0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).mean())


def _text_mask_tile(text: str, font_path: str, font_size: int, stroke_width: int) -> Image.Image:
    font = ImageFont.truetype(font_path, font_size)
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 0))
    bbox = probe.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    tile = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(tile)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255)
    tight = tile.getbbox()
    if tight is None:
        raise RuntimeError(f"Font rendered empty text for {text!r}")
    return tile.crop(tight)


def _draw_tile(base: Image.Image, text: str, tile_mask: Image.Image, px: int, py: int, style: RenderStyle, target: Tuple[int, int, int, int]) -> Rendered:
    lum = _local_luminance(base, target)
    fill = (255, 255, 255) if lum < 0.52 else (0, 0, 0)
    stroke = (0, 0, 0) if fill[0] > 128 else (255, 255, 255)
    full_mask = Image.new("L", base.size, 0)
    full_mask.paste(tile_mask, (px, py), tile_mask)
    coords = occupied_patch_coords(full_mask, 14 if base.size == (224, 224) else max(1, base.width // 16))
    result = base.copy().convert("RGB")
    solid = Image.new("RGB", tile_mask.size, fill)
    # A visible opposite-polarity outline is already encoded in tile_mask. RGB
    # uses the fill polarity; the exact glyph support mask remains authoritative
    # for patch supervision.
    if style.stroke_width > 0:
        font = ImageFont.truetype(style.font_path, style.font_size)
        probe = ImageDraw.Draw(Image.new("L", (8, 8), 0))
        bbox = probe.textbbox((0, 0), text, font=font, stroke_width=style.stroke_width)
        rgba = Image.new("RGBA", (max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])), (0,0,0,0))
        d = ImageDraw.Draw(rgba)
        d.text((-bbox[0], -bbox[1]), text, font=font, fill=fill+(255,), stroke_width=style.stroke_width, stroke_fill=stroke+(255,))
        ab = rgba.getchannel("A").getbbox()
        if ab is not None:
            rgba = rgba.crop(ab)
        if rgba.size == tile_mask.size:
            result.paste(rgba.convert("RGB"), (px, py), tile_mask)
        else:
            result.paste(solid, (px, py), tile_mask)
    else:
        result.paste(solid, (px, py), tile_mask)
    ink_bbox = full_mask.getbbox()
    if ink_bbox is None:
        raise RuntimeError("empty glyph mask")
    return Rendered(result, full_mask, coords, style, tuple(map(int, ink_bbox)))


def render_patch_fitted_word(base: Image.Image, text: str, placement: Placement, patch_size: int, font_paths: Sequence[str], rng: random.Random, safety: int = 1, style_hint: Optional[RenderStyle] = None) -> Rendered:
    """Render an exact one-patch or two-horizontal-patch word.

    If style_hint is supplied, font/size/stroke are held fixed; this is used to
    make supportive and adversarial copies of the same literal word optically
    matched while allowing local black/white polarity to maintain visibility.
    """
    if not font_paths:
        raise RuntimeError("Tiny-patch training requires at least one usable TrueType font")
    W, H = base.size
    if W % patch_size or H % patch_size:
        raise ValueError(f"Image size {base.size} not divisible by patch={patch_size}")
    if placement.geometry == "one_patch":
        x0, y0 = placement.col * patch_size, placement.row * patch_size
        target = (x0+safety, y0+safety, x0+patch_size-safety, y0+patch_size-safety)
        expected = [(placement.row, placement.col)]
    elif placement.geometry == "two_patch_x":
        x0, y0 = placement.col * patch_size, placement.row * patch_size
        target = (x0+safety, y0+safety, x0+2*patch_size-safety, y0+patch_size-safety)
        expected = [(placement.row, placement.col), (placement.row, placement.col+1)]
    else:
        raise ValueError(placement.geometry)
    tw_max, th_max = target[2]-target[0], target[3]-target[1]
    render_text = text.upper()

    if style_hint is not None:
        styles = [style_hint]
    else:
        font_order = list(font_paths)
        rng.shuffle(font_order)
        styles = []
        for stroke_width in (1, 0):
            for font_path in font_order:
                for font_size in range(min(18, patch_size+4), 4, -1):
                    styles.append(RenderStyle(font_path, font_size, stroke_width))

    for style in styles:
        try:
            tile_mask = _text_mask_tile(render_text, style.font_path, style.font_size, style.stroke_width)
        except Exception:
            continue
        mw, mh = tile_mask.size
        if mw > tw_max or mh > th_max:
            continue
        py = target[1] + (th_max-mh)//2
        if placement.geometry == "one_patch":
            px_candidates = [target[0] + (tw_max-mw)//2]
        else:
            boundary = (placement.col+1)*patch_size
            ideal = int(round(boundary - mw/2.0))
            offsets = [0]
            for delta in range(1, patch_size):
                offsets.extend((-delta, delta))
            px_candidates = [ideal+d for d in offsets]
        for px in px_candidates:
            if px < target[0] or px+mw > target[2]:
                continue
            full_mask = Image.new("L", base.size, 0)
            full_mask.paste(tile_mask, (px, py), tile_mask)
            if occupied_patch_coords(full_mask, patch_size) != expected:
                continue
            rendered = _draw_tile(base, render_text, tile_mask, px, py, style, target)
            rendered.coords = expected
            return rendered
    raise RuntimeError(f"Could not render {text!r} as {placement.geometry} with requested style")


def render_two_patch_matched(base: Image.Image, text: str, placement: Placement, patch_size: int, one_patch: Rendered, safety: int = 1) -> Rendered:
    if placement.geometry != "two_patch_x":
        raise ValueError("matched renderer expects two_patch_x")
    return render_patch_fitted_word(base, text, placement, patch_size, [one_patch.style.font_path], random.Random(0), safety=safety, style_hint=one_patch.style)


def draw_patch_grid(image: Image.Image, mask: Image.Image, patch_size: int) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for x in range(0, out.width+1, patch_size):
        draw.line((x,0,x,out.height), fill=(255,0,0), width=1)
    for y in range(0, out.height+1, patch_size):
        draw.line((0,y,out.width,y), fill=(255,0,0), width=1)
    for row, col in occupied_patch_coords(mask, patch_size):
        x0, y0 = col*patch_size, row*patch_size
        draw.rectangle((x0,y0,x0+patch_size-1,y0+patch_size-1), outline=(0,255,0), width=2)
    return out


class TinyPatchImageNetBuilder:
    """Online synthetic tiny-text source built from existing ImageNet packet clean rows."""

    def __init__(self, digital_root: Path, wnid_json: Path, split: str, image_size: int, patch_size: int, font_paths: Sequence[str], words: str = "all", geometry_weights: Sequence[float] = (0.35, 0.25, 0.40), patch_safety: int = 1):
        self.digital_root = Path(digital_root)
        self.wnid_json = Path(wnid_json)
        self.split = str(split)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.font_paths = [str(x) for x in font_paths if Path(str(x)).is_file()]
        if not self.font_paths:
            raise RuntimeError("TinyPatchImageNetBuilder found no usable TrueType fonts")
        self.patch_safety = int(patch_safety)
        gw = [max(0.0, float(x)) for x in geometry_weights]
        if len(gw) != 3 or sum(gw) <= 0:
            raise ValueError(f"geometry_weights must be three non-negative values, got {geometry_weights}")
        self.geometry_weights = gw

        lookup_obj = json.loads(self.wnid_json.read_text(encoding="utf-8"))
        self.lookup: Dict[str,str] = {str(k): str(v) for k,v in lookup_obj.items()}
        taxonomy = build_short_concept_taxonomy(self.lookup)
        requested = [x.strip().lower() for x in str(words).split(",") if x.strip()]
        if requested == ["all"] or not requested:
            requested = sorted(taxonomy)
        unknown = sorted(set(requested)-set(taxonomy))
        if unknown:
            raise ValueError(f"Unknown tiny-patch words {unknown}; available={sorted(taxonomy)}")
        self.taxonomy = {w:taxonomy[w] for w in requested if taxonomy[w].wnids}

        manifest = self.digital_root / "manifests" / f"{self.split}.jsonl"
        rows = read_jsonl(manifest)
        self.by_group: Dict[str,List[Dict[str,Any]]] = defaultdict(list)
        for row in rows:
            self.by_group[str(row.get("group_id"))].append(row)
        self.group_meta: Dict[str,Dict[str,Any]] = {}
        for gid, grows in self.by_group.items():
            wnid = parse_group_wnid(gid, grows)
            clean_rows = [r for r in grows if str(r.get("relation")) == "none"]
            if not wnid or wnid not in self.lookup or not clean_rows:
                continue
            self.group_meta[gid] = {
                "wnid":wnid,
                "class_label":self.lookup[wnid],
                "domain_words":_taxonomy_membership(wnid,self.taxonomy),
                "clean_rows":clean_rows,
            }
        all_gids = sorted(self.group_meta)
        self.support_pools: Dict[str,List[str]] = {}
        self.adversarial_pools: Dict[str,List[str]] = {}
        for word,spec in self.taxonomy.items():
            support=[g for g in all_gids if self.group_meta[g]["wnid"] in spec.wnids]
            cross=[]
            fallback=[]
            for g in all_gids:
                meta=self.group_meta[g]
                if meta["wnid"] in spec.wnids:
                    continue
                toks=set(re.findall(r"[a-z0-9]+",meta["class_label"].lower()))
                if word in toks:
                    continue
                fallback.append(g)
                other=meta["domain_words"]
                if any(self.taxonomy[ow].domain != spec.domain for ow in other):
                    cross.append(g)
            self.support_pools[word]=support
            self.adversarial_pools[word]=cross or fallback
        self.words=[w for w in sorted(self.taxonomy) if self.support_pools[w] and self.adversarial_pools[w]]
        if not self.words:
            raise RuntimeError("No usable tiny-patch ImageNet concepts found")

    @property
    def group_count(self) -> int:
        return len(self.group_meta)

    @property
    def concept_count(self) -> int:
        return len(self.words)

    def coverage(self) -> Dict[str,Any]:
        return {
            "split": self.split,
            "groups": self.group_count,
            "concepts": self.concept_count,
            "words": self.words,
            "support_groups": {w:len(self.support_pools[w]) for w in self.words},
            "adversarial_groups": {w:len(self.adversarial_pools[w]) for w in self.words},
        }

    def _clean(self, gid: str, rng: random.Random) -> Image.Image:
        row = rng.choice(self.group_meta[gid]["clean_rows"])
        rel = str(row["image_relpath"])
        return crop_like_training(Image.open(self.digital_root/rel).convert("RGB"), self.image_size)

    @staticmethod
    def _pseudo(word: str, rng: random.Random) -> str:
        vowels="aeiou"; consonants="bcdfghjklmnpqrstvwxyz"; out=[]
        for ch in word:
            pool=vowels if ch.lower() in vowels else consonants
            choices=[x for x in pool if x != ch.lower()]
            out.append(rng.choice(choices or list(pool)))
        value="".join(out)
        return value if value != word else value[:-1]+("x" if value[-1] != "x" else "z")

    def _choose_geometry(self, rng: random.Random) -> str:
        return rng.choices(["one_patch","two_patch_x_matched","two_patch_x_large"], weights=self.geometry_weights, k=1)[0]

    def sample(self, rng: random.Random) -> TinyPatchPacketData:
        # Rendering can rarely fail for an unusually wide pseudoword/font. Retry
        # with another concept/style rather than emitting malformed occupancy.
        for attempt in range(32):
            word=rng.choice(self.words)
            support_gid=rng.choice(self.support_pools[word])
            adv_candidates=[g for g in self.adversarial_pools[word] if g != support_gid]
            if not adv_candidates:
                continue
            adv_gid=rng.choice(adv_candidates)
            support=self._clean(support_gid,rng)
            adversarial=self._clean(adv_gid,rng)
            pseudo=self._pseudo(word,rng)
            row=rng.randint(2,max(2,self.grid-3))
            col=rng.randint(2,max(2,self.grid-4))
            one_place=Placement("one_patch",row,col)
            two_place=Placement("two_patch_x",row,col)
            geometry=self._choose_geometry(rng)
            try:
                if geometry == "one_patch":
                    sup_word=render_patch_fitted_word(support,word,one_place,self.patch_size,self.font_paths,rng,self.patch_safety)
                    adv_word=render_patch_fitted_word(adversarial,word,one_place,self.patch_size,self.font_paths,rng,self.patch_safety,style_hint=sup_word.style)
                    adv_pseudo=render_patch_fitted_word(adversarial,pseudo,one_place,self.patch_size,self.font_paths,rng,self.patch_safety)
                elif geometry == "two_patch_x_matched":
                    # First derive the maximal one-patch style, then translate the
                    # SAME style over one x-boundary on both semantic roles.
                    sup_template=render_patch_fitted_word(support,word,one_place,self.patch_size,self.font_paths,rng,self.patch_safety)
                    sup_word=render_two_patch_matched(support,word,two_place,self.patch_size,sup_template,self.patch_safety)
                    adv_word=render_patch_fitted_word(adversarial,word,two_place,self.patch_size,self.font_paths,rng,self.patch_safety,style_hint=sup_template.style)
                    pseudo_template=render_patch_fitted_word(adversarial,pseudo,one_place,self.patch_size,self.font_paths,rng,self.patch_safety)
                    adv_pseudo=render_two_patch_matched(adversarial,pseudo,two_place,self.patch_size,pseudo_template,self.patch_safety)
                else:
                    sup_word=render_patch_fitted_word(support,word,two_place,self.patch_size,self.font_paths,rng,self.patch_safety)
                    adv_word=render_patch_fitted_word(adversarial,word,two_place,self.patch_size,self.font_paths,rng,self.patch_safety,style_hint=sup_word.style)
                    adv_pseudo=render_patch_fitted_word(adversarial,pseudo,two_place,self.patch_size,self.font_paths,rng,self.patch_safety)
            except Exception:
                continue

            support_label=self.group_meta[support_gid]["class_label"]
            adv_label=self.group_meta[adv_gid]["class_label"]
            captions=[
                f"a photo of a {support_label}",                 # 0
                f"<notext> a photo of a {support_label}",        # 1
                f"a photo of a {adv_label}",                     # 2
                f"<notext> a photo of a {adv_label}",            # 3
                f"<text> {word}",                                # 4
                f"<text> {pseudo}",                              # 5
                f'a photo of a {support_label} with the text "{word}"', # 6
                f'a photo of a {adv_label} with the text "{word}"',     # 7
                f'a photo of a {adv_label} with the text "{pseudo}"',   # 8
                "<text> <null>",                                  # 9
                f"a photo of a {word}",                          # 10 hard semantic negative on adversarial image
            ]
            # Images: clean/supportive literal/clean adversarial/adversarial literal/adversarial pseudo.
            positive_pairs=[]
            # Object semantics stay stable within each source image family.
            for img in (0,1):
                positive_pairs.extend(((img,0),(img,1)))
            for img in (2,3,4):
                positive_pairs.extend(((img,2),(img,3)))
            # Literal support is independent of semantic correctness.
            positive_pairs.extend(((1,4),(3,4),(4,5),(1,6),(3,7),(4,8),(0,9),(2,9)))

            return TinyPatchPacketData(
                images=[support,sup_word.image,adversarial,adv_word.image,adv_pseudo.image],
                masks=[Image.new("L",support.size,0),sup_word.mask,Image.new("L",adversarial.size,0),adv_word.mask,adv_pseudo.mask],
                mask_weights=[1.0]*5,
                present_targets=[0.0,1.0,0.0,1.0,1.0],
                readable_targets=[0.0,1.0,0.0,1.0,1.0],
                captions=captions,
                positive_pairs=positive_pairs,
                # Same literal word must rank above both clean images and the
                # pseudoword image in BOTH supportive and adversarial contexts.
                source_triplets=[
                    (4,1,[0,2,4]),
                    (4,3,[0,2,4]),
                    (5,4,[0,1,2,3]),
                    (9,0,[1,3,4]),
                    (9,2,[1,3,4]),
                ],
                auto_triplets=[(3,2,10)],
                invariance_pairs=[(0,1),(2,3),(2,4)],
                metadata={
                    "source":"imagenet_tiny",
                    "concept_word":word,
                    "pseudo_text":pseudo,
                    "geometry":geometry,
                    "support_group_id":support_gid,
                    "support_wnid":self.group_meta[support_gid]["wnid"],
                    "support_label":support_label,
                    "adversarial_group_id":adv_gid,
                    "adversarial_wnid":self.group_meta[adv_gid]["wnid"],
                    "adversarial_label":adv_label,
                    "word_style":{
                        "font_size":sup_word.style.font_size,
                        "stroke_width":sup_word.style.stroke_width,
                        "font_name":Path(sup_word.style.font_path).name,
                    },
                    "word_occupied_patches":sup_word.coords,
                    "pseudo_occupied_patches":adv_pseudo.coords,
                    "adversarial_any_negative":f"a photo of a {word}",
                },
            )
        raise RuntimeError("Tiny-patch generator failed after 32 retries")

    def save_preview_grids(self, out_dir: Path, count: int, seed: int) -> List[Path]:
        out_dir=Path(out_dir)
        out_dir.mkdir(parents=True,exist_ok=True)
        rng=random.Random(int(seed))
        saved=[]
        for i in range(max(0,int(count))):
            packet=self.sample(rng)
            # Alternate supportive/adversarial/pseudoword previews.
            idx=(1,3,4)[i%3]
            preview=draw_patch_grid(packet.images[idx],packet.masks[idx],self.patch_size)
            name=f"tiny_patch_{i:02d}_{packet.metadata['concept_word']}_{packet.metadata['geometry']}_{idx}.png"
            path=out_dir/name
            preview.save(path)
            saved.append(path)
        return saved
