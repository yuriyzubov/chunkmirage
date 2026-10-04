var e=`from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from chunkmirage.core import ArrayInfo

log = logging.getLogger("chunkmirage")


class Op(BaseModel):
    """A block-wise operation.

    Subclass, set \`\`name\`\`, declare parameters as pydantic fields, and implement \`\`apply\`\`.

    * \`\`halo\`\`: voxels of upstream context needed on every side (int or per-axis tuple).
      The framework reads the padded region, calls \`\`apply\`\` on it, and crops the result.
    * \`\`cache\`\`: whether this stage's output chunks should be memoized. Turn on for
      expensive stages (inference) so cheap downstream tweaks (threshold) are free. A
      pipeline can override it per op: \`\`{"op": "gaussian", "sigma": 4, "cache": true}\`\`.
    * \`\`output_dtype\`\` / \`\`output_info\`\`: describe the result; default is unchanged.
    * \`\`output_kind\`\`: what the result's values are (\`\`ArrayInfo.kind\`\`): \`\`image\`\`,
      \`\`label\`\` or \`\`mask\`\`; \`\`None\`\` (the default) if the op does not say.
    * \`\`cache_token()\`\`: what the result depends on beyond the parameters (a file's
      modification time, weights replaced in place); part of the op's identity.
    * \`\`slots\`\`: at most this many \`\`apply\`\` calls of this op at once, process-wide (a GPU,
      a memory-hungry step), queued finest level first; \`\`None\`\` (the default): no limit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ClassVar[str] = ""
    halo: ClassVar[int | tuple[int, ...]] = 0
    cache: ClassVar[bool] = False
    # packages its apply imports beyond numpy (the browser engine loads them up front)
    packages: ClassVar[tuple[str, ...]] = ()
    output_kind: ClassVar[str | None] = None
    slots: ClassVar[int | None] = None
    _cache: bool | None = PrivateAttr(None)  # this op's own setting, over the class's

    @property
    def cached(self) -> bool:
        """Whether this op's output is memoized: its own setting, else its class's."""
        return self.cache if self._cache is None else self._cache

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        h = self.halo
        return tuple(h) if isinstance(h, (tuple, list)) else (int(h),) * ndim

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(in_dtype)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(dtype=self.output_dtype(info.dtype), kind=self.output_kind)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def input_voxel_size(self) -> tuple[float, ...] | None:
        """The voxel size this op must read, along the data's last axes and in the source's
        units: a model trained at one resolution. The pipeline then runs it on one level (the
        source's at that size, else one resampled to it), caches what it makes, and makes the
        coarser levels by downsampling that. \`\`None\`\` (the default): it runs on every level."""
        return None

    def for_level(self, info: ArrayInfo) -> Op:
        """This op as it runs on a scale level described by \`\`info\`\` (the stage's output, on
        the input's grid). Ops that work in physical units override it to take the level's
        voxel size: a slope in degrees needs the pixel spacing, which doubles from level to
        level. The default is the op itself."""
        return self

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        """Like \`\`apply\`\` but told where \`\`block\`\` sits (\`\`box\`\` = its halo-padded extent in
        voxels of this scale level). Override when the result must depend on position, e.g. to
        make per-chunk labels globally unique. Default delegates to \`\`apply\`\`."""
        return self.apply(block)

    # --- identity / serialization -------------------------------------------------
    def spec(self) -> dict[str, Any]:
        spec = {"op": self.name, **self.model_dump(mode="json")}
        if self._cache is not None:
            spec["cache"] = self._cache
        return spec

    def cache_token(self) -> str | None:
        """What this op's output depends on outside its parameters, as a string that changes
        when that does: a weights file's modification time, a model version. \`\`None\`\` (the
        default): nothing. It is part of \`\`digest\`\`, so a pipeline built after it changes has
        new stage keys and new links, and viewers refetch; \`\`DatasetRegistry.refresh\`\`
        rebuilds a served dataset to read it again."""
        return None

    def digest(self) -> str:
        """Identity of what the op computes: whether it is cached does not change that."""
        identity = {"op": self.name, **self.model_dump(mode="json")}
        token = self.cache_token()
        if token is not None:
            identity["@cache_token"] = str(token)
        payload = json.dumps(identity, sort_keys=True, default=str).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


_REGISTRY: dict[str, type[Op]] = {}
_ENTRYPOINTS_LOADED = False


def register(cls: type[Op]) -> type[Op]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a class-level \`name\`")
    _REGISTRY[cls.name] = cls
    return cls


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in entry_points(group="chunkmirage.ops"):
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Op):
                _REGISTRY.setdefault(cls.name or ep.name, cls)
        except Exception:  # noqa: BLE001 - a broken plugin must not take the server down
            log.warning("op plugin %r (%s) failed to load; skipped", ep.name, ep.value, exc_info=True)


def get_op(name: str) -> type[Op]:
    _load_entrypoints()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown op {name!r}; known: {sorted(_REGISTRY)}") from None


def list_ops() -> dict[str, dict]:
    _load_entrypoints()
    return {
        name: {
            "halo": "dynamic" if isinstance(cls.halo, property) else cls.halo,
            "cache": cls.cache,
            "slots": cls.slots,
            "doc": (cls.__doc__ or "").strip(),
            "schema": cls.model_json_schema(),
        }
        for name, cls in sorted(_REGISTRY.items())
    }


def op_from_spec(spec: dict[str, Any] | Op) -> Op:
    if isinstance(spec, Op):
        return spec
    spec = dict(spec)
    name = spec.pop("op")
    cache = spec.pop("cache", None)
    op = get_op(name)(**spec)
    if cache is not None:
        op._cache = bool(cache)
    return op


def ops_from_specs(specs: Sequence[dict[str, Any] | Op]) -> list[Op]:
    return [op_from_spec(s) for s in specs]
`,t=`"""Ops over the channels of a \`\`stack://\`\` source: results that need two images at once.

Such an op takes a block whose first axis is the stack's channel axis and returns one
without it; the pipeline reads every channel and pads only the spatial axes by the halo.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Contacts(Op):
    """Contact sites between two structures, the first two channels of a \`\`stack://\`\` source:
    the voxels within \`\`radius\`\` of both. A mask; follow with \`\`label\`\` to colour and
    size-filter the sites."""

    name = "contacts"
    packages = ("scipy",)
    radius: float = Field(
        3.0,
        gt=0,
        le=32,
        description="Reach, in voxels: a voxel is a contact site when both structures lie within "
        "this distance of it (Euclidean). The halo is radius + 1.",
    )
    distance: float | None = Field(
        None,
        gt=0,
        description="Reach in the data's units (nm, say), instead of radius: each level counts "
        "it in its own voxels, so a contact means the same at every zoom (a level whose voxels "
        "are bigger than it keeps the voxels in both structures).",
    )
    a_low: float = Field(
        128.0,
        description="Values at or above this in the first channel are the first structure: 128 "
        "for a uint8 probability map, 1 for a segmentation.",
    )
    b_low: float = Field(
        128.0,
        description="The same threshold for the second channel, the second structure.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.radius) + 1

    def for_level(self, info: ArrayInfo) -> Op:
        """With \`\`distance\`\`, the radius in this level's voxels (its finest spatial axis)."""
        if self.distance is None:
            return self
        op = self.model_copy(update={"radius": self.distance / min(info.voxel_size[-3:])})
        op._cache = self._cache
        return op

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        if info.ndim < 4 or info.axes[0] != "c" or info.shape[0] < 2:
            raise ValueError(
                "contacts needs a source whose first axis is a channel axis holding two "
                f"structures, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:],
            dtype=self.output_dtype(info.dtype),
            chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:],
            units=info.units[1:],
            axes=info.axes[1:],
            translation=info.translation[1:],
            kind="mask",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = block[0] >= self.a_low
        b = block[1] >= self.b_low
        return (self._within_reach(a) & self._within_reach(b)).astype(np.uint8)

    def _within_reach(self, mask: np.ndarray) -> np.ndarray:
        """The voxels within \`\`radius\`\` of \`\`mask\`\`: the distance to it, thresholded."""
        from scipy.ndimage import distance_transform_edt

        if not mask.any():
            return np.zeros(mask.shape, dtype=bool)
        if mask.all():
            return np.ones(mask.shape, dtype=bool)
        return distance_transform_edt(~mask) <= self.radius


@register
class NormalizedDifference(Op):
    """\`\`(a - b) / (a + b)\`\` of two channels of a \`\`stack://\`\` source (float32): the indices
    remote sensing reads plants, water and burn scars from (NDVI is near infrared and red,
    NBR near and shortwave infrared). With \`\`minus\`\`, a second pair's index is subtracted
    from the first's: a change between two dates. Burn severity (dNBR) is the NBR before a
    fire minus the NBR after, the before and after images' bands stacked as four channels.
    Where a band is zero or less, or the two sum to under \`\`floor\`\` (water), the index is NaN."""

    name = "normalized_difference"
    pair: list[int] = Field(
        [0, 1], min_length=2, max_length=2, description="The channels a and b, counted from the first."
    )
    minus: list[int] | None = Field(
        None,
        min_length=2,
        max_length=2,
        description="Two more channels, whose index is subtracted from the first pair's: [2, 3] "
        "for the after image of a stack of before and after.",
    )
    offset: float = Field(
        0.0,
        description="Added to every channel first, to make reflectances of stored numbers: "
        "Sentinel-2 since 2022 stores them plus 1000 (offset -1000).",
    )
    nodata: float | None = Field(
        None, description="A stored value meaning no data (Sentinel-2: 0): the index there is NaN."
    )
    floor: float = Field(
        0.0,
        ge=0,
        description="Where a pair's two values sum to less than this (after offset), its index is "
        "noise, a ratio of near zeros (water reflects almost no infrared), and is NaN.",
    )
    output_kind = "image"

    def _channels(self) -> list[int]:
        return [*self.pair, *(self.minus or [])]

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        need = max(self._channels()) + 1
        if info.ndim < 3 or info.axes[0] != "c" or info.shape[0] < need:
            raise ValueError(
                f"normalized_difference needs a source whose first axis holds {need} or more "
                f"channels, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:], dtype=np.dtype("float32"), chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:], units=info.units[1:], axes=info.axes[1:],
            translation=info.translation[1:], kind="image",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        def index(a: int, b: int) -> np.ndarray:
            x, y = (block[c].astype(np.float32) + np.float32(self.offset) for c in (a, b))
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (x - y) / (x + y)
            out[(x <= 0) | (y <= 0) | (x + y < self.floor)] = np.nan  # positive, and not near zero
            if self.nodata is not None:
                out[(block[a] == self.nodata) | (block[b] == self.nodata)] = np.nan
            return out

        out = index(*self.pair)
        return out - index(*self.minus) if self.minus else out
`,n=`"""Core value types shared by sources, ops, frontends and the server.

Conventions
-----------
* Arrays are handled in numpy C order. For a 3-D volume that means axes \`\`(z, y, x)\`\`
  with \`\`x\`\` varying fastest in memory. Frontends that use the opposite convention
  (N5, precomputed) reverse axis lists when emitting metadata / parsing chunk keys.
* \`\`Box\`\` is half-open \`\`[start, stop)\`\` in voxel coordinates of a given scale level.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Half-open voxel box \`\`[start, stop)\`\` in C-order axes."""

    start: tuple[int, ...]
    stop: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.start) != len(self.stop):
            raise ValueError("start and stop must have same length")

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.start, self.stop))

    @property
    def empty(self) -> bool:
        return any(s <= 0 for s in self.shape)

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop))

    def pad(self, halo: Sequence[int]) -> Box:
        return Box(
            tuple(a - h for a, h in zip(self.start, halo)),
            tuple(b + h for b, h in zip(self.stop, halo)),
        )

    def clip(self, shape: Sequence[int]) -> Box:
        return Box(
            tuple(min(max(a, 0), s) for a, s in zip(self.start, shape)),
            tuple(min(max(b, 0), s) for b, s in zip(self.stop, shape)),
        )

    def relative_to(self, other: Box) -> Box:
        """Express this box in coordinates whose origin is \`\`other.start\`\`."""
        return Box(
            tuple(a - o for a, o in zip(self.start, other.start)),
            tuple(b - o for b, o in zip(self.stop, other.start)),
        )

    @staticmethod
    def from_chunk(index: Sequence[int], chunk_shape: Sequence[int]) -> Box:
        start = tuple(i * c for i, c in zip(index, chunk_shape))
        stop = tuple(s + c for s, c in zip(start, chunk_shape))
        return Box(start, stop)


KINDS = (None, "image", "label", "mask")


def kind_for_dtype(dtype) -> str | None:
    """What values of \`\`dtype\`\` most likely are when nothing says (a stored array): \`\`mask\`\`
    for booleans, \`\`label\`\` for integers of 32 bits or more (segment ids; intensities are
    rarely stored that wide), else unknown. Labels and masks are resampled by nearest voxel
    and downsampled by their most common value, never averaged."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return "mask"
    if dtype.kind in "iu" and dtype.itemsize >= 4:
        return "label"
    return None


@dataclass(frozen=True)
class ArrayInfo:
    """Static description of one scale level of a chunked array (C-order axes). \`\`kind\`\` is
    what its values are, for viewers choosing how to show it: \`\`image\`\` (intensities),
    \`\`label\`\` (segment ids) or \`\`mask\`\` (inside or not); \`\`None\`\` if unknown."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunk_shape: tuple[int, ...]
    voxel_size: tuple[float, ...]
    units: tuple[str, ...]
    axes: tuple[str, ...]
    translation: tuple[float, ...] = field(default=None)  # type: ignore[assignment]
    kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "chunk_shape", tuple(int(s) for s in self.chunk_shape))
        n = len(self.shape)
        if self.translation is None:
            object.__setattr__(self, "translation", (0.0,) * n)
        for name in ("chunk_shape", "voxel_size", "units", "axes", "translation"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length {n}, got {getattr(self, name)}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_grid(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunk_shape))

    def chunk_box(self, index: Sequence[int], clip: bool = True) -> Box:
        box = Box.from_chunk(index, self.chunk_shape)
        return box.clip(self.shape) if clip else box

    def chunks_covering(self, box: Box) -> Iterator[tuple[int, ...]]:
        """Yield chunk indices whose extent intersects \`\`box\`\` (assumed already clipped)."""
        if box.empty:
            return
        lo = [a // c for a, c in zip(box.start, self.chunk_shape)]
        hi = [(b - 1) // c for b, c in zip(box.stop, self.chunk_shape)]
        for idx in np.ndindex(*[h - lo_i + 1 for lo_i, h in zip(lo, hi)]):
            yield tuple(int(lo_i + i) for lo_i, i in zip(lo, idx))

    def with_(self, **changes) -> ArrayInfo:
        from dataclasses import replace

        return replace(self, **changes)

    def rescaled(self, voxel_size: Sequence[float]) -> ArrayInfo:
        """This array's extent on voxels of \`\`voxel_size\`\` along its last axes (as many as
        given): the shape rounded up, chunks as they were (in voxels), and the translation
        moved so each voxel's position is its centre, as OME-Zarr's is (a voxel twice as big
        covers two, its centre half an old voxel on)."""
        k = len(voxel_size)
        old, new = self.voxel_size[-k:], tuple(float(v) for v in voxel_size)
        shape = tuple(math.ceil(n * o / v - 1e-9) for n, o, v in zip(self.shape[-k:], old, new))
        shift = tuple(t + (v - o) / 2 for t, o, v in zip(self.translation[-k:], old, new))
        return self.with_(
            shape=self.shape[:-k] + shape,
            voxel_size=self.voxel_size[:-k] + new,
            translation=self.translation[:-k] + shift,
        )

    @staticmethod
    def default_axes(ndim: int) -> tuple[str, ...]:
        names = ("t", "c", "z", "y", "x")
        return names[-ndim:] if ndim <= 5 else tuple(f"d{i}" for i in range(ndim))
`,r=`"""Neighbourhood filters (most need scipy). These exercise the halo machinery."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Gaussian(Op):
    """Gaussian blur (smoothing). Reduces noise before thresholding; larger sigma = blurrier."""

    name = "gaussian"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        description="Blur width in voxels (standard deviation). 1 removes pixel noise; 3-5 merges small structures.",
    )
    truncate: float = Field(
        3.0,
        ge=1,
        le=6,
        description="Kernel radius in units of sigma. Rarely needs changing; 3 keeps 99.7% of the kernel. "
        "Determines the halo: ceil(sigma × truncate) voxels of neighbouring data are read on each side.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.truncate)

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            block.astype(np.float32), self.sigma, truncate=self.truncate, mode="nearest"
        )


@register
class Uniform(Op):
    """Box (mean) filter: each voxel becomes the average of a size³ cube around it."""

    name = "uniform"
    packages = ("scipy",)
    size: int = Field(
        3,
        ge=1,
        le=31,
        description="Edge length of the averaging cube, in voxels (odd values are centred).",
    )

    @property
    def halo(self):  # type: ignore[override]
        return self.size // 2 + 1

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        return uniform_filter(block.astype(np.float32), self.size, mode="nearest")


@register
class Diff(Op):
    """Change along one axis: each voxel minus the one \`\`lag\`\` steps before it on \`\`axis\`\`
    (float32). Along time, what changed since the frame or day before: a hurricane's cold
    wake in sea temperature, a flare brightening the sun. The first \`\`lag\`\` steps of the
    array compare against its first."""

    name = "diff"
    axis: int = Field(
        0,
        ge=0,
        description="The axis to difference along, counted from the first (0: time in a t, y, x series).",
    )
    lag: int = Field(1, ge=1, le=64, description="How many steps back to compare with.")

    @property
    def halo(self):  # type: ignore[override]
        return self.lag

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        if self.axis >= ndim:
            raise ValueError(f"diff axis={self.axis}: the data has {ndim} axes")
        return tuple(self.lag if a == self.axis else 0 for a in range(ndim))

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = self.axis
        b = block.astype(np.float32)
        out = np.zeros_like(b)
        now, before = [slice(None)] * b.ndim, [slice(None)] * b.ndim
        now[a], before[a] = slice(self.lag, None), slice(None, -self.lag)
        out[tuple(now)] = b[tuple(now)] - b[tuple(before)]
        return out


@register
class Gradient(Op):
    """Rate of change along each of \`\`axes\`\`, one channel each on a new leading \`\`c\`\` axis
    (float32), per unit of the axes (a level's voxel size): central differences. It returns
    only the interior it can compute, one voxel less on each side of those axes, as a valid
    convolution (or a model) does. Where temperature changes fastest at sea, ocean fronts;
    on terrain, the slope's components; in a volume, edges and their direction. With
    \`\`sigma\`\`, smoothed along the same axes first (a Gaussian derivative), so noise a voxel or
    two across does not point every voxel its own way."""

    name = "gradient"
    packages = ("scipy",)
    sigma: float = Field(
        0.0,
        ge=0,
        le=16,
        description="Smoothing first, in voxels along the axes differentiated (0: none). The halo "
        "grows to 1 + ceil(3 × sigma).",
    )
    axes: list[int] | None = Field(
        None,
        description="The axes to differentiate along, counted from the first (1, 2: latitude and "
        "longitude of a time, lat, lon series); default the last three, or all if fewer.",
    )
    _voxel: tuple[float, ...] | None = PrivateAttr(None)

    def _along(self, ndim: int) -> list[int]:
        axes = list(range(max(0, ndim - 3), ndim)) if self.axes is None else list(self.axes)
        if not axes or any(not 0 <= a < ndim for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"gradient axes={self.axes}: the data has {ndim} axes")
        return axes

    @property
    def halo(self):  # type: ignore[override]
        return 1 + math.ceil(3 * self.sigma)

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        along = self._along(ndim)
        return tuple(self.halo if a in along else 0 for a in range(ndim))

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        n = len(self._along(info.ndim))
        return info.with_(
            shape=(n, *info.shape), chunk_shape=(n, *info.chunk_shape), dtype=np.dtype("float32"),
            voxel_size=(1.0, *info.voxel_size), units=("", *info.units), axes=("c", *info.axes),
            translation=(0.0, *info.translation), kind="image",
        )

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._voxel = tuple(float(v) for v in info.voxel_size)
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32)
        along = self._along(b.ndim)
        voxel = self._voxel[-b.ndim:] if self._voxel else (1.0,) * b.ndim
        if self.sigma > 0:
            from scipy.ndimage import gaussian_filter

            b = gaussian_filter(b, [self.sigma if a in along else 0 for a in range(b.ndim)], truncate=3, mode="nearest")
        h = self.halo
        inner = [slice(h, -h) if a in along else slice(None) for a in range(b.ndim)]
        out = []
        for a in along:
            hi, lo = list(inner), list(inner)
            hi[a], lo[a] = slice(h + 1, b.shape[a] - h + 1), slice(h - 1, b.shape[a] - h - 1)
            out.append((b[tuple(hi)] - b[tuple(lo)]) / np.float32(2 * voxel[a]))
        return np.stack(out)


@register
class Downsample(Op):
    """Coarser voxels: each block of \`\`factor\`\` voxels becomes one, their mean, or for labels
    and masks their most common value. The grid changes with it: voxels \`\`factor\`\` times
    bigger, the shape divided (rounded up), each voxel's position the centre of its block.
    A pipeline makes the coarser levels of an op with an input voxel size this way."""

    name = "downsample"
    factor: list[int] = Field(
        [2, 2, 2],
        description="Voxels per output voxel along each of the data's last axes (z, y, x): "
        "2, 2, 2 halves each; 1 keeps an axis as it is.",
    )
    mode: str = Field(
        "auto",
        pattern="^(auto|mean|mode)$",
        description="mean of each block; mode, its most common value (labels, masks); auto: "
        "mode for labels and masks, mean for anything else.",
    )
    _mode: str = PrivateAttr("mean")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        f = self.factor
        if not f or len(f) > info.ndim or any(int(v) < 1 for v in f):
            raise ValueError(f"downsample factor={f}: one whole number of 1 or more per axis, of {info.ndim}")
        return info.rescaled([v * k for v, k in zip(info.voxel_size[-len(f) :], f)])

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._mode = self.mode if self.mode != "auto" else ("mode" if info.kind in ("label", "mask") else "mean")
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        f, k = [int(v) for v in self.factor], len(self.factor)
        lead, space = block.shape[:-k], block.shape[-k:]
        whole = [-(-n // v) * v for n, v in zip(space, f)]
        if list(space) != whole:  # the array's last block along an axis: its last voxel repeated
            block = np.pad(block, [(0, 0)] * len(lead) + [(0, w - n) for w, n in zip(whole, space)], "edge")
        shape = list(lead) + [x for w, v in zip(whole, f) for x in (w // v, v)]
        blocks = block.reshape(shape)
        inner = tuple(len(lead) + 2 * i + 1 for i in range(k))  # the axes within each block
        if self._mode == "mean":  # in the data's own type, integers rounded
            mean = blocks.mean(axis=inner, dtype=np.float64)
            return (np.rint(mean) if np.issubdtype(block.dtype, np.integer) else mean).astype(block.dtype)
        order = [a for a in range(blocks.ndim) if a not in inner] + list(inner)
        g = blocks.transpose(order).reshape(*lead, *(w // v for w, v in zip(whole, f)), -1)
        counts = (g[..., :, None] == g[..., None, :]).sum(-1)  # how often each value of a block occurs in it
        return np.take_along_axis(g, counts.argmax(-1)[..., None], -1)[..., 0]
`,i=`"""Ops run back to back on one block: what a pipeline stage computes for a chunk.

Imports nothing but numpy and the ops, so the browser engine runs this same code in
Pyodide: the page reads the padded input, this computes the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op


def _ratio(a: ArrayInfo, b: ArrayInfo, k: int) -> tuple[float, ...]:
    """Input voxels per output voxel along the last \`\`k\`\` axes, from \`\`a\`\`'s grid to \`\`b\`\`'s."""
    return tuple(float(w) / float(v) for v, w in zip(a.voxel_size[-k:], b.voxel_size[-k:]))


CHANNEL_AXES = ("c", "channel", "channels")


def _gridded(info: ArrayInfo) -> int:
    """How many trailing axes are chunked and padded: those after the last channel axis.
    A channel axis, and any axis before it, is read whole, so an op may change its length
    (select channels, combine them) or drop it."""
    last = max((a for a, name in enumerate(info.axes) if name in CHANNEL_AXES), default=-1)
    return info.ndim - last - 1


def _infos(info: ArrayInfo, ops: Sequence[Op]) -> tuple[list[ArrayInfo], int]:
    """Each op's output, and how many trailing axes every op keeps: those no op drops or
    adds, and none of them a channel axis."""
    infos, kept = [info], _gridded(info)
    for op in ops:
        infos.append(op.output_info(infos[-1]))
        kept = min(kept, _gridded(infos[-1]))
    return infos, kept


def scale(info: ArrayInfo, ops: Sequence[Op]) -> tuple[float, ...]:
    """Input voxels each output voxel spans along the axes \`\`ops\`\` keep: 1 for ops that keep
    the grid. Only a stage's first op may change it (\`\`downsample\`\`, or a model that reads 8
    nm voxels and writes 16 nm ones)."""
    infos, kept = _infos(info, ops)
    for i, (a, b) in enumerate(zip(infos[1:], infos[2:]), 1):
        if any(abs(r - 1) > 1e-9 for r in _ratio(a, b, kept)):
            raise ValueError(f"op {ops[i].name!r} changes the voxel size: only a stage's first op may")
    return _ratio(infos[0], infos[-1], kept) if ops else (1.0,) * kept


def plan(info: ArrayInfo, ops: Sequence[Op]) -> tuple[ArrayInfo, int, tuple[int, ...]]:
    """What \`\`ops\`\` make of input \`\`info\`\`: the output's info (its chunks still the input's),
    how many leading axes they consume (the channels of a \`\`stack://\`\` source, which are
    read whole), and the halo the axes they keep need, in input voxels: the sum of the ops',
    those after a first op that changes the grid counted in its output's voxels (\`\`scale\`\`).
    An op may also add leading axes (a model's channels; read whole too): the output's axes
    are those it added, then those kept. Planned on the finest level, the halo is the widest
    any level needs."""
    infos, kept = _infos(info, ops)
    out, r = infos[-1], scale(info, ops)
    lead = info.ndim - kept
    # each op as it runs on this level (an op in physical units counts its halo in its voxels)
    halos = [op.for_level(out).halo_for(kept) for op in ops]
    halo = tuple(
        int(np.ceil((halos[0][a] if halos else 0) + r[a] * sum(h[a] for h in halos[1:]) - 1e-9))
        for a in range(kept)
    )
    return out, lead, halo


def _whole(values, what: str) -> tuple[int, ...]:
    out = tuple(int(round(v)) for v in values)
    if any(abs(v - o) > 1e-6 for v, o in zip(values, out)):
        raise ValueError(f"{what} {tuple(values)} is not whole voxels of the input: choose chunks the voxel ratio divides")
    return out


def input_box(info: ArrayInfo, out_box: Box, lead: int, halo: Sequence[int],
              scale: Sequence[float] | None = None) -> Box:
    """The input to read for output box \`\`out_box\`\`: every leading axis, the kept ones
    (the last \`\`len(halo)\`\` of the output's), on the input's grid (\`\`scale\`\` input voxels
    to each output voxel), padded."""
    k = len(halo)
    r = scale or (1.0,) * k
    start = _whole([o * f for o, f in zip(out_box.start[-k:], r)], "an output chunk's start")
    stop = _whole([o * f for o, f in zip(out_box.stop[-k:], r)], "an output chunk's end")
    padded = Box(start, stop).pad(halo)
    return Box((0,) * lead + padded.start, info.shape[:lead] + padded.stop)


def run(ops: Sequence[Op], block: np.ndarray, in_box: Box, out_box: Box, out: ArrayInfo,
        halo: Sequence[int] | None = None, scale: Sequence[float] | None = None):
    """\`\`ops\`\` on \`\`block\`\` (the input over \`\`in_box\`\`), cropped to \`\`out_box\`\`. \`\`halo\`\` and
    \`\`scale\`\` are \`\`plan\`\`'s and \`\`scale\`\`'s: the halo's length is how many trailing axes the
    ops keep (by default all the output's). Each op returns its block as it came, or shaved
    by its own halo on every side (as a valid convolution does), and may drop leading axes
    or add some; a first op that changes the grid returns its output's voxels over either."""
    k = len(halo) if halo is not None else min(in_box.ndim, out.ndim)
    r = tuple(scale) if scale is not None else (1.0,) * k
    space = Box(in_box.start[-k:], in_box.stop[-k:])  # where the block sits in the kept axes
    box = in_box  # and in all the axes it has
    for i, op in enumerate(ops):
        level_op = op.for_level(out)
        result = level_op.apply_at(block, box)
        h = level_op.halo_for(k)
        got = result.shape[-k:] if result.ndim >= k else None
        if i == 0 and any(abs(f - 1) > 1e-9 for f in r):  # onto the output's grid
            for cut in (h, (0,) * k):  # shaved by its halo first, as a model's output is
                lo = [(a + c) / f for a, c, f in zip(space.start, cut, r)]
                hi = [(b - c) / f for b, c, f in zip(space.stop, cut, r)]
                if all(abs(v - round(v)) < 1e-6 for v in lo + hi) and got == tuple(round(b - a) for a, b in zip(lo, hi)):
                    space = Box(tuple(round(v) for v in lo), tuple(round(v) for v in hi))
                    break
            else:
                raise ValueError(
                    f"op {op.name!r} returned {result.shape} from {block.shape}: on a grid {r} times "
                    f"coarser, the block's voxels or those within its halo {h}"
                )
        elif got != block.shape[-k:]:
            if got != tuple(s - 2 * a for s, a in zip(block.shape[-k:], h)):
                raise ValueError(
                    f"op {op.name!r} changed block shape {block.shape} -> {result.shape}: it may "
                    f"drop or add leading axes, and shave its halo {h} off the last {k}"
                )
            space = space.pad([-a for a in h])  # a valid convolution: it returned the interior
        lead = result.shape[: result.ndim - k]  # leading axes are read whole
        same = lead == block.shape[: block.ndim - k]  # as they came, or made anew
        box = Box(box.start[: len(lead)] if same else (0,) * len(lead),
                  box.stop[: len(lead)] if same else tuple(lead))
        box = Box(box.start + space.start, box.stop + space.stop)
        block = result
    if block.ndim != out.ndim:
        raise ValueError(f"ops left a block of {block.ndim} axes for an output of {out.ndim}")
    crop = Box(out_box.start[-k:], out_box.stop[-k:]).relative_to(space)
    lead = Box(out_box.start[: out.ndim - k], out_box.stop[: out.ndim - k])
    return np.asarray(block[lead.slices() + crop.slices()], dtype=out.dtype)


def pad_edge(data: np.ndarray, box: Box, shape: Sequence[int]) -> np.ndarray:
    """\`\`data\`\`, read over \`\`box\`\` clipped to an array of \`\`shape\`\`, extended to all of
    \`\`box\`\` by repeating the nearest voxel inside (\`\`Source.read_padded(edge=True)\`\`)."""
    clipped = box.clip(shape)
    if clipped == box:
        return data
    inner = clipped.relative_to(box)
    return np.pad(data, [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)], "edge")
`,a=`"""Ops: pure functions on blocks with declared halo, output dtype and cacheability."""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.combine import Contacts, NormalizedDifference
from chunkmirage.ops.filters import Diff, Downsample, Gaussian, Gradient, Uniform
from chunkmirage.ops.pointwise import Cast, Scale, Threshold
from chunkmirage.ops.segment import DoG, Label, Morphology, Spots
from chunkmirage.ops.terrain import Hillshade, Slope

__all__ = [
    "Cast",
    "Contacts",
    "Diff",
    "Downsample",
    "Gradient",
    "DoG",
    "Label",
    "Morphology",
    "NormalizedDifference",
    "Gaussian",
    "Hillshade",
    "Op",
    "Scale",
    "Slope",
    "Spots",
    "Threshold",
    "Uniform",
    "get_op",
    "list_ops",
    "op_from_spec",
    "register",
]
`,o=`from __future__ import annotations

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class Threshold(Op):
    """Binary mask: voxels with \`\`low <= value < high\`\` become \`\`value\`\`, everything else 0."""

    name = "threshold"
    output_kind = "mask"
    low: float = Field(
        0.0,
        description="Lower bound (inclusive), in the source's intensity units. Voxels at or above it pass.",
    )
    high: float | None = Field(
        None,
        description="Upper bound (exclusive). Leave empty for no upper bound. Use it to select an intensity band.",
    )
    value: int = Field(
        1,
        ge=1,
        le=255,
        description="Label written for voxels that pass (1 shows as one segment in Neuroglancer).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        mask = block >= self.low
        if self.high is not None:
            mask &= block < self.high
        return mask.astype(np.uint8) * np.uint8(self.value)


@register
class Cast(Op):
    """Convert to another data type, e.g. float32 → uint8 for viewers that need integers."""

    name = "cast"
    dtype: str = Field(
        "uint8",
        description="Target numpy dtype name: uint8, uint16, uint32, uint64, int16, float32, ...",
    )
    clip: bool = Field(
        True,
        description="Clip values to the target integer range first (avoids wrap-around, e.g. 300 → 255 not 44).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype(self.dtype)

    def output_info(self, info):
        return info.with_(dtype=self.output_dtype(info.dtype))  # still what it was: labels stay labels

    def apply(self, block: np.ndarray) -> np.ndarray:
        out = np.dtype(self.dtype)
        if self.clip and np.issubdtype(out, np.integer):
            ii = np.iinfo(out)
            block = np.clip(block, ii.min, ii.max)
        return block.astype(out)


@register
class Scale(Op):
    """Linear intensity rescale \`\`value * factor + offset\`\` (output is float32)."""

    name = "scale"
    factor: float = Field(1.0, description="Multiply every voxel by this (contrast).")
    offset: float = Field(0.0, description="Then add this (brightness).")

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        return block.astype(np.float32) * np.float32(self.factor) + np.float32(self.offset)
`,s=`"""Showcase ops beyond pointwise: multi-scale filtering, morphology and labelling (need scipy).

These demonstrate what a viewer shader cannot do: neighbourhood context (halos), binary
morphology, and connected-component labelling with size filtering.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class DoG(Op):
    """Difference of Gaussians: enhances blob-like structures of a chosen size, suppresses background."""

    name = "dog"
    output_kind = "image"
    packages = ("scipy",)
    sigma: float = Field(
        2.0, gt=0, description="Size of structures to enhance, in voxels (smaller blur)."
    )
    ratio: float = Field(
        1.6,
        gt=1,
        le=4,
        description="Larger blur = sigma × ratio. 1.6 approximates a Laplacian of Gaussian.",
    )
    gain: float = Field(
        4.0, gt=0, description="Multiply the difference so the result uses the 0..255 range."
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.ratio * 3)

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        b = block.astype(np.float32)
        d = gaussian_filter(b, self.sigma, mode="nearest") - gaussian_filter(
            b, self.sigma * self.ratio, mode="nearest"
        )
        return np.clip(d * self.gain + 128, 0, 255).astype(np.uint8)


@register
class Morphology(Op):
    """Binary morphology on a mask: remove specks (open), fill holes (close), shrink or grow."""

    name = "morphology"
    output_kind = "mask"
    packages = ("scipy",)
    operation: str = Field(
        "open",
        pattern="^(open|close|erode|dilate)$",
        description="open = erode then dilate (removes small objects); close = dilate then erode (fills small gaps); erode; dilate.",
    )
    radius: int = Field(
        2, ge=1, le=16, description="Radius of the spherical structuring element, in voxels."
    )

    @property
    def halo(self):  # type: ignore[override]
        return 2 * self.radius + 1

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

        r = self.radius
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = (zz * zz + yy * yy + xx * xx) <= r * r
        mask = block > 0
        fn = {
            "open": binary_opening,
            "close": binary_closing,
            "erode": binary_erosion,
            "dilate": binary_dilation,
        }[self.operation]
        return fn(mask, structure=ball).astype(np.uint8)


@register
class Label(Op):
    """Connected components of a mask, coloured as segments. Labels are unique per chunk, so one
    object spanning several chunks gets several colours; that is the honest per-chunk preview."""

    name = "label"
    output_kind = "label"
    packages = ("scipy",)
    min_size: int = Field(
        0,
        ge=0,
        description="Drop components smaller than this many voxels (counted within the chunk + halo).",
    )
    connectivity: int = Field(
        1,
        ge=1,
        le=3,
        description="1 = faces only (6-connected), 2 = +edges (18), 3 = +corners (26).",
    )

    @property
    def halo(self):  # type: ignore[override]
        # Some context so size filtering near chunk borders sees more of each object.
        return 4 if self.min_size else 0

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(3, self.connectivity)
        labels, n = label(block > 0, structure=structure)
        labels = labels.astype(np.uint32)
        if self.min_size and n:
            sizes = np.bincount(labels.ravel(), minlength=n + 1)
            keep = sizes >= self.min_size
            keep[0] = False
            labels = np.where(keep[labels], labels, 0).astype(np.uint32)
        return labels

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        labels = self.apply(block)
        # Offset labels by a salt derived from the chunk position so colours differ between
        # chunks (labels stay < 2**16 per chunk; the salt occupies the high 16 bits).
        z, y, x = (int(v) for v in box.start[-3:])
        salt = np.uint32(((z * 73856093) ^ (y * 19349663) ^ (x * 83492791)) & 0x7FFF) << np.uint32(
            16
        )
        return np.where(labels > 0, labels + salt, 0).astype(np.uint32)


@register
class Spots(Op):
    """Bright diffraction-limited spots, such as single mRNA molecules in smFISH or EASI-FISH:
    a difference of Gaussians, its local maxima above \`\`threshold\`\`, each drawn as a small
    ball. A spot's id comes from its position in the volume, so one spot keeps its id and
    colour whichever chunk finds it."""

    name = "spots"
    output_kind = "label"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        le=8,
        description="Spot size, in voxels along y and x: the smaller blur of the difference of "
        "Gaussians (the larger is 1.6 times it). About the spot's standard deviation.",
    )
    sigma_z: float = Field(
        0.6,
        gt=0,
        le=8,
        description="The same along z, in z voxels: smaller than sigma when z voxels are "
        "coarser, as in most light microscopy.",
    )
    threshold: float = Field(
        10.0,
        ge=0,
        description="Least difference-of-Gaussians response a spot needs, in the image's "
        "intensity units: higher finds fewer, brighter spots.",
    )
    separation: int = Field(
        2,
        ge=1,
        le=8,
        description="Spots closer than this, in y-x voxels, are one: the radius of the local "
        "maximum search.",
    )
    radius: int = Field(
        1,
        ge=0,
        le=8,
        description="Radius of the ball drawn for each spot, in y-x voxels (0 marks one voxel).",
    )

    def _z(self, v: float) -> int:
        """\`\`v\`\` y-x voxels in z voxels, as far as sigma_z / sigma says."""
        return int(math.ceil(v * self.sigma_z / self.sigma))

    @property
    def halo(self):  # type: ignore[override]
        xy = math.ceil(3 * 1.6 * self.sigma) + self.separation + self.radius
        z = math.ceil(3 * 1.6 * self.sigma_z) + self._z(self.separation) + self._z(self.radius)
        return (z, xy, xy)

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def output_info(self, info):
        if info.ndim != 3:
            raise ValueError(
                f"spots finds spots in a z, y, x volume, not axes {info.axes}: pin the other "
                'axes with the spec\\'s select, e.g. {"c": 1, "t": 0} (--select c=1,t=0)'
            )
        return super().output_info(info)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - apply_at is used
        return self.apply_at(block, None)

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, grey_dilation, maximum_filter

        b = block.astype(np.float32)
        s = (self.sigma_z, self.sigma, self.sigma)
        dog = gaussian_filter(b, s, mode="nearest") - gaussian_filter(
            b, tuple(1.6 * v for v in s), mode="nearest"
        )
        sz, sxy = self._z(self.separation), self.separation
        peaks = (dog == maximum_filter(dog, size=(2 * sz + 1, 2 * sxy + 1, 2 * sxy + 1))) & (
            dog >= self.threshold
        )
        out = np.zeros(b.shape, dtype=np.uint32)
        where = np.nonzero(peaks)
        if not len(where[0]):
            return out
        origin = np.asarray(box.start[-3:] if box is not None else (0, 0, 0), dtype=np.int64)
        z, y, x = (w.astype(np.int64) + o for w, o in zip(where, origin))
        h = (z * 73856093) ^ (y * 19349663) ^ (x * 83492791)
        out[where] = (h % 0xFFFFFFFE + 1).astype(np.uint32)
        if self.radius:
            rz, r = self._z(self.radius), self.radius
            zz, yy, xx = np.ogrid[-rz : rz + 1, -r : r + 1, -r : r + 1]
            ball = (zz / max(rz, 1)) ** 2 * (rz > 0) + (yy * yy + xx * xx) / (r * r) <= 1
            out = grey_dilation(out, footprint=ball)
        return out
`,c=`"""Terrain from elevation: slope and hillshade, the first things one computes from a digital
elevation model. Both work on the last two axes (\`\`y, x\`\`, rows down the image) in the
level's own pixel spacing, so a coarse level's slope is the slope of its coarser grid."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


class _Terrain(Op):
    halo = 1
    _spacing: tuple[float, float] = PrivateAttr((1.0, 1.0))

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._spacing = (float(info.voxel_size[-2]), float(info.voxel_size[-1]))
        op._cache = self._cache
        return op

    def _gradient(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d(elevation)/dy and /dx, per unit of the axes (y counting down the image)."""
        z = block.astype(np.float64) * self.z_factor
        dy, dx = np.gradient(z, *self._spacing, axis=(-2, -1))
        return dy, dx


@register
class Slope(_Terrain):
    """Slope of an elevation model in degrees (0 flat, 90 a cliff), from the level's pixel
    spacing; output float32, NaN where the elevation is."""

    name = "slope"
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing (1 when both are metres).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


@register
class Hillshade(_Terrain):
    """Shaded relief: the brightness of the terrain lit by a distant sun at \`\`azimuth\`\` and
    \`\`altitude\`\` (local illumination only: no shadows cast across the terrain). Output
    uint8, 1 (unlit) to 255 (facing the sun), 0 where the elevation is NaN."""

    name = "hillshade"
    azimuth: float = Field(
        315.0,
        ge=0,
        le=360,
        description="Direction the sun shines from, degrees clockwise from the top of the image.",
    )
    altitude: float = Field(
        45.0, gt=0, le=90, description="Height of the sun above the horizon, in degrees."
    )
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing; above 1 exaggerates relief.",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        az, alt = np.radians(self.azimuth), np.radians(self.altitude)
        # the surface normal (-dz/dx, dz/dy up the image, 1) against the sun's direction
        sun = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
        lit = (-dx * sun[0] + dy * sun[1] + sun[2]) / np.sqrt(dx * dx + dy * dy + 1.0)
        out = (1 + 254 * np.clip(lit, 0.0, 1.0)).round()
        return np.where(np.isfinite(lit), out, 0).astype(np.uint8)
`,l=`"""Byte-bounded LRU cache for numpy chunks (and encoded bytes)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


def _nbytes(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 1


class LRUCache(Generic[T]):
    """Thread-safe LRU keyed by any hashable, bounded by total bytes.

    A single shared instance is normally used for every pipeline stage; keys embed the
    stage hash so that changing a parameter downstream never evicts upstream entries
    except through normal LRU pressure.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[Hashable, T] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> T | None:
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                self.misses += 1
                return None
            self._data[key] = value
            self.hits += 1
            return value

    def put(self, key: Hashable, value: T) -> None:
        size = _nbytes(value)
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= _nbytes(self._data.pop(key))
            self._data[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes and self._data:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= _nbytes(evicted)

    def invalidate(self, prefix: Hashable | None = None) -> int:
        """Drop entries; if \`\`prefix\`\` is given, only tuple keys starting with it."""
        with self._lock:
            if prefix is None:
                n = len(self._data)
                self._data.clear()
                self._bytes = 0
                return n
            doomed = [k for k in self._data if isinstance(k, tuple) and k[:1] == (prefix,)]
            for k in doomed:
                self._bytes -= _nbytes(self._data.pop(k))
            return len(doomed)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "entries": len(self._data),
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
`,u=`from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import KINDS, ArrayInfo, Box


class Source(ABC):
    """Read-only, random-access array. Everything upstream of an op is a \`\`Source\`\`."""

    @property
    @abstractmethod
    def info(self) -> ArrayInfo: ...

    @abstractmethod
    def read(self, box: Box) -> np.ndarray:
        """Return data for \`\`box\`\` (must lie within \`\`info.shape\`\`), shape == box.shape."""

    def read_padded(self, box: Box, fill=0, *, edge: bool = False) -> np.ndarray:
        """Read \`\`box\`\` even if it pokes outside the array. Out-of-range voxels are \`\`fill\`\`,
        or with \`\`edge\`\` the nearest voxel inside (so a filter sees no step at the border)."""
        clipped = box.clip(self.info.shape)
        if clipped == box:
            return self.read(box)
        if edge and not clipped.empty:
            inner = clipped.relative_to(box)
            pad = [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)]
            return np.pad(self.read(clipped), pad, mode="edge")
        out = np.full(box.shape, fill, dtype=self.info.dtype)
        if not clipped.empty:
            out[clipped.relative_to(box).slices()] = self.read(clipped)
        return out

    def cache_key(self) -> str:
        """Stable identity used to build cache keys for downstream stages."""
        return f"{type(self).__name__}:{id(self)}"


class KindSource(Source):
    """\`\`inner\`\` with its values said to be \`\`kind\`\` (\`\`ArrayInfo.kind\`\`), whatever it
    guessed: a uint32 image, or labels stored as uint16."""

    def __init__(self, inner: Source, kind: str | None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.inner = inner
        self._info = inner.info.with_(kind=kind)
        self._key = f"kind={kind}:{inner.cache_key()}"  # resampled and downsampled differently

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        return self.inner.read(box)


class ChunkedSource(Source):
    """A source materialized chunk-by-chunk via \`\`compute_chunk\`\` and memoized in an LRU.

    \`\`read(box)\`\` gathers the covering chunks (from cache or freshly computed) and slices.
    This is the single mechanism behind *both* raw-source caching and per-stage caching:
    every pipeline stage is exposed to the next one as a \`\`ChunkedSource\`\`.
    """

    def __init__(
        self,
        info: ArrayInfo,
        compute_chunk: Callable[[tuple[int, ...]], np.ndarray],
        cache: LRUCache | None,
        key: str,
    ):
        self._info = info
        self._compute = compute_chunk
        self._cache = cache
        self._key = key
        # In-flight computations keyed by chunk index: concurrent requests for the same chunk
        # (a viewer retrying, or two clients) wait for one computation instead of duplicating it.
        self._inflight: dict[tuple[int, ...], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def chunk(self, index: Sequence[int]) -> np.ndarray:
        """Full (edge-clipped) chunk \`\`index\`\` as an array of shape \`\`info.chunk_box(index).shape\`\`."""
        index = tuple(int(i) for i in index)
        if self._cache is None:
            return self._compute_checked(index)
        while True:
            hit = self._cache.get((self._key, index))
            if hit is not None:
                return hit
            with self._inflight_lock:
                event = self._inflight.get(index)
                if event is None:
                    event = self._inflight[index] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                event.wait()
                continue  # the owner has cached it (or failed); re-check the cache
            try:
                data = self._compute_checked(index)
                self._cache.put((self._key, index), data)
                return data
            finally:
                with self._inflight_lock:
                    self._inflight.pop(index, None)
                event.set()

    def _compute_checked(self, index: tuple[int, ...]) -> np.ndarray:
        data = np.ascontiguousarray(self._compute(index))
        expected = self._info.chunk_box(index).shape
        if data.shape != expected:
            raise ValueError(
                f"{self._key}: compute_chunk{index} returned shape {data.shape}, expected {expected}"
            )
        return data

    def read(self, box: Box) -> np.ndarray:
        info = self._info
        # Fast path: the box is exactly one chunk.
        idx = tuple(a // c for a, c in zip(box.start, info.chunk_shape))
        if info.chunk_box(idx) == box:
            return self.chunk(idx)
        out = np.empty(box.shape, dtype=info.dtype)
        for cidx in info.chunks_covering(box):
            cbox = info.chunk_box(cidx)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, cbox.start)),
                tuple(min(a, b) for a, b in zip(box.stop, cbox.stop)),
            )
            if inter.empty:
                continue
            data = self.chunk(cidx)
            out[inter.relative_to(box).slices()] = data[inter.relative_to(cbox).slices()]
        return out


class MultiscaleSource:
    """An ordered list of \`\`Source\`\` levels, s0 = full resolution. \`\`shader\`\` is a
    Neuroglancer shader the viewer should show the data with (its channel axis then being
    a shader channel), for sources whose channels mean something particular."""

    def __init__(self, levels: Sequence[Source], name: str = "", shader: str | None = None):
        if not levels:
            raise ValueError("need at least one level")
        self.levels = list(levels)
        self.name = name
        self.shader = shader

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, i: int) -> Source:
        return self.levels[i]

    def __iter__(self):
        return iter(self.levels)

    def cache_key(self) -> str:
        return "|".join(lvl.cache_key() for lvl in self.levels)

    def level_for(self, voxel_size: Sequence[float], rtol: float = 0.01) -> tuple[int, bool]:
        """The level to read data at \`\`voxel_size\`\` from (its last axes, in the source's
        units), and whether it is at that size: one that is (within \`\`rtol\`\`), else the
        coarsest finer on every axis (resampled down, never up), else level 0."""
        want = np.asarray(voxel_size, dtype=float)
        best = 0
        for i, lvl in enumerate(self.levels):
            v = np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)
            if np.allclose(v, want, rtol=rtol, atol=0):
                return i, True
            if np.all(v <= want * (1 + rtol)):
                best = i
        return best, False

    def nearest_level(self, voxel_size: Sequence[float]) -> int:
        """The level whose voxel size (its last axes) is nearest \`\`voxel_size\`\`, by ratio
        (the sum over axes of the size's log ratio); the finer of two as near."""
        want = np.log(np.asarray(voxel_size, dtype=float))
        dist = [
            float(np.abs(np.log(np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)) - want).sum())
            for lvl in self.levels
        ]
        return min(range(len(dist)), key=lambda i: (round(dist[i], 9), i))
`,d=`"""Procedurally generated sources: arbitrarily large, nothing on disk, exact at every scale.

URL form (accepted by \`\`open_source\`\`)::

    synthetic://blobs?shape=4096,4096,4096&chunk=64,64,64&levels=5&seed=0&voxel_size=8
    synthetic://noise?...        fractal (fBm) value noise
    synthetic://shells?...       hollow spheres (membrane-like)
    synthetic://julia?...        3-D slice of a quaternion Julia set
    synthetic://mandelbulb?shape=1073741824,1073741824,1073741824   the Mandelbulb, to zoom into
    synthetic://blobs+noise?...  sum of kinds

Every voxel is a deterministic function of its world coordinate, so level *i* is simply
the same function sampled with a 2**i voxel spacing: \`\`s1[z, y, x] == s0[2z, 2y, 2x]\`\`.
That makes the pyramid free and exactly self-consistent, unlike averaging. The one
exception is \`\`mandelbulb\`\`, which iterates more the finer the level (as fractal zoomers
do: detail a coarse level could not show needs more iterations to resolve), so a coarse
level's boundary is a little fuller than a fine one's. Its coordinates are float64, so an
array 2**30 voxels across, 23 levels deep, still resolves its finest voxels.

Generation is numpy on ~260k voxels per chunk; numpy releases the GIL for those
operations, so the server's threadpool already computes chunks on all cores.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

_KINDS = ("blobs", "noise", "julia", "shells", "mandelbulb")
F32 = np.float32


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint32 hash of integer lattice coordinates -> float32 in [0, 1)."""
    h = (
        np.asarray(x).astype(np.uint32) * np.uint32(73856093)
        ^ np.asarray(y).astype(np.uint32) * np.uint32(19349663)
        ^ np.asarray(z).astype(np.uint32) * np.uint32(83492791)
        ^ np.uint32((seed * 2654435761) & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h.astype(F32) * F32(1.0 / 4294967296.0)


def _lattice(z, y, x, cell: float):
    """Per-voxel lattice cell indices (int) and fractional offsets, plus the covered ranges."""
    fz, fy, fx = (np.asarray(v, dtype=F32) / F32(cell) for v in (z, y, x))
    iz, iy, ix = (np.floor(f).astype(np.int64) for f in (fz, fy, fx))
    tz, ty, tx = fz - iz, fy - iy, fx - ix
    rng = [(int(i.min()) - 1, int(i.max()) + 2) for i in (iz, iy, ix)]  # +2: room for corner+1
    return (iz, iy, ix), (tz, ty, tx), rng


def _grid(rng, seed):
    """Hash values on the small lattice grid covering \`\`rng\`\` (inclusive lows, exclusive highs)."""
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    return _hash3(gx, gy, gz, seed), (z0, y0, x0)


def _value_noise(z, y, x, cell: float, seed: int) -> np.ndarray:
    """Trilinear lattice noise. Hashes the (small) lattice once, then gathers 8 corners."""
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z, y, x, cell)
    vals, (z0, y0, x0) = _grid(rng, seed)
    tz, ty, tx = (t * t * (3 - 2 * t) for t in (tz, ty, tx))  # smoothstep
    iz, iy, ix = iz - z0, iy - y0, ix - x0
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    for dz in (0, 1):
        wz = tz if dz else 1 - tz
        for dy in (0, 1):
            wy = ty if dy else 1 - ty
            for dx in (0, 1):
                wx = tx if dx else 1 - tx
                out += wz * wy * wx * vals[iz + dz, iy + dy, ix + dx]
    return out


def _fbm(z, y, x, seed: int, base_cell: float = 256.0, octaves: int = 2) -> np.ndarray:
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    amp, total, cell = 1.0, 0.0, base_cell
    for o in range(octaves):
        out += F32(amp) * _value_noise(z, y, x, cell, seed + o)
        total += amp
        amp *= 0.5
        cell /= 2
    return out / F32(total)


# Blobs / shells: one lattice cell holds \`per_cell\` random spheres whose influence is cut off
# at \`reach\` <= cell/2. Two evaluation strategies with identical results: loop over cells and
# evaluate each sphere on its footprint (cheap when a chunk spans few cells, i.e. fine
# levels), or gather per voxel from the 8 nearest cells (bounded cost at coarse levels; a
# sphere reaching at most cell/2 can only touch voxels in the 2x2x2 cells nearest to them).
_CELL_LOOP_MAX = 216  # switch strategy above this many cells


def _cell_params(gz, gy, gx, seed: int, k: int, cell: float, kind: str):
    s = seed * 131 + k * 7
    u = [_hash3(gx, gy, gz, s + i) for i in range(5)]
    if kind == "blobs":
        center = ((gz + u[0]) * cell, (gy + u[1]) * cell, (gx + u[2]) * cell)
        radius = 8 + u[3] * (cell / 6 - 8)  # 8 .. cell/6, so reach = 3r <= cell/2
        amp = 90 + u[4] * 130
        reach = 3 * radius
    else:  # shells: centred, larger, one per cell; reach = r + 10 <= cell/2
        center = tuple((g + 0.5 + (uu - 0.5) * 0.4) * cell for g, uu in zip((gz, gy, gx), u[:3]))
        radius = cell * (0.22 + 0.15 * u[3])
        amp = np.full_like(radius, 200)
        reach = radius + 10
    f = lambda a: np.asarray(a, dtype=F32)  # noqa: E731
    return tuple(f(c) for c in center), f(radius), f(amp), f(reach)


def _sphere_field(d2, radius, amp, reach, kind):
    inside = d2 <= reach * reach
    if kind == "blobs":
        return inside * amp * np.exp(-d2 / (2 * radius * radius))
    d = np.sqrt(d2)
    return inside * (amp * np.exp(-((d - radius) ** 2) / F32(2 * 2.5**2)) + F32(40) * (d < radius))


def _spheres(z, y, x, seed: int, kind: str) -> np.ndarray:
    cell, per_cell = (128.0, 2) if kind == "blobs" else (160.0, 1)
    z32, y32, x32 = (np.asarray(v, dtype=F32) for v in (z, y, x))
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z32, y32, x32, cell)
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    n_cells = gz.size
    # nearest-cell base index per voxel: floor(f - 0.5), relative to the grid origin
    nz, ny, nx = (i - (t < 0.5) for i, t in ((iz, tz), (iy, ty), (ix, tx)))
    for k in range(per_cell):
        (cz, cy, cx), rad, amp, reach = _cell_params(gz, gy, gx, seed, k, cell, kind)
        if n_cells <= _CELL_LOOP_MAX:
            # loop over cells, evaluate each sphere only on its footprint
            for i in np.ndindex(gz.shape):
                c0, c1, c2, r, a, rc = cz[i], cy[i], cx[i], rad[i], amp[i], reach[i]
                sl = []
                for coord, c in zip((z32.ravel(), y32.ravel(), x32.ravel()), (c0, c1, c2)):
                    lo = int(np.searchsorted(coord, c - rc))
                    hi = int(np.searchsorted(coord, c + rc, side="right"))
                    if hi <= lo:
                        break
                    sl.append(slice(lo, hi))
                if len(sl) < 3:
                    continue
                zz, yy, xx = z32[sl[0]], y32[:, sl[1]], x32[:, :, sl[2]]
                d2 = (zz - c0) ** 2 + (yy - c1) ** 2 + (xx - c2) ** 2
                out[tuple(sl)] += _sphere_field(d2, r, a, rc, kind)
        else:
            # gather from the 8 nearest cells for every voxel
            for dz in (0, 1):
                jz = nz - z0 + dz
                for dy in (0, 1):
                    jy = ny - y0 + dy
                    for dx in (0, 1):
                        jx = nx - x0 + dx
                        jz_, jy_, jx_ = np.broadcast_arrays(jz, jy, jx)
                        d2 = (
                            (z32 - cz[jz_, jy_, jx_]) ** 2
                            + (y32 - cy[jz_, jy_, jx_]) ** 2
                            + (x32 - cx[jz_, jy_, jx_]) ** 2
                        )
                        out += _sphere_field(
                            d2, rad[jz_, jy_, jx_], amp[jz_, jy_, jx_], reach[jz_, jy_, jx_], kind
                        )
    return out


def _julia(z, y, x, seed: int, scale: float, max_iter: int = 14) -> np.ndarray:
    """Escape-time of a quaternion Julia set sampled on a 3-D slice (w=0). Bright = slow escape."""
    rng = np.random.default_rng(seed)
    c = (rng.uniform(-0.8, 0.4, 4) * np.array([1, 1, 1, 0.3])).astype(F32)
    qx, qy, qz = np.broadcast_arrays(
        *(np.asarray(v, dtype=F32) / F32(scale) - F32(1.4) for v in (x, y, z))
    )
    qx, qy, qz = qx.copy(), qy.copy(), qz.copy()
    qw = np.zeros_like(qx)
    count = np.zeros(qx.shape, dtype=F32)
    alive = np.ones(qx.shape, dtype=bool)
    for _ in range(max_iter):
        nx = qx * qx - qy * qy - qz * qz - qw * qw + c[0]
        ny = 2 * qx * qy + c[1]
        nz = 2 * qx * qz + c[2]
        nw = 2 * qx * qw + c[3]
        alive &= (nx * nx + ny * ny + nz * nz + nw * nw) < 16
        qx, qy, qz, qw = (np.where(alive, v, F32(0)) for v in (nx, ny, nz, nw))
        count += alive
    return F32(255.0 / max_iter) * count


BULB = 1.25  # the Mandelbulb's array spans [-BULB, BULB] on each axis
BULB_POWER = 8.0


def _mandelbulb(start, stop, step: int, full_shape) -> np.ndarray:
    """The power-8 Mandelbulb's escape time: 4 x the (smooth) iteration at which a point
    leaves radius 2, so a colour stays a colour from level to level, 255 for points that
    have not left (inside). The array spans [-1.25, 1.25] on every axis, in float64 from
    integer indices; a level of voxels \`\`v\`\` across iterates 10 + 3 log2(256 / v) times
    (10 for the whole bulb on 256 voxels, a few more for every halving)."""
    across = BULB * 2 / max(full_shape)  # the finest voxel, in bulb units
    iters = int(np.clip(10 + 3 * np.log2(max(full_shape) / (256 * step)), 10, 80))
    axes = [
        ((np.arange(a, b, dtype=np.float64) + 0.5) * step) * across - BULB
        for a, b in zip(start, stop)
    ]
    pz, py, px = np.meshgrid(*axes, indexing="ij")
    x, y, z = px.copy(), py.copy(), pz.copy()
    out = np.full(px.shape, 255.0)
    alive = np.ones(px.shape, dtype=bool)
    for i in range(iters):
        r = np.sqrt(x * x + y * y + z * z)
        gone = alive & (r > 2)
        if gone.any():  # smooth escape: the fraction of an iteration past radius 2
            out[gone] = 4 * (i + 1 - np.log(np.log(r[gone]) / np.log(2)) / np.log(BULB_POWER))
            alive &= ~gone
        if not alive.any():
            break
        ra = r[alive]
        theta = BULB_POWER * np.arccos(np.clip(z[alive] / np.maximum(ra, 1e-300), -1, 1))
        phi = BULB_POWER * np.arctan2(y[alive], x[alive])
        rn = ra**BULB_POWER
        x[alive] = rn * np.sin(theta) * np.cos(phi) + px[alive]
        y[alive] = rn * np.sin(theta) * np.sin(phi) + py[alive]
        z[alive] = rn * np.cos(theta) + pz[alive]
    return np.where(alive, 255, np.clip(out, 0, 254)).astype(np.float32)


def generate(
    kinds: tuple[str, ...], full_shape: tuple[int, ...], level: int, seed: int, start, stop
) -> np.ndarray:
    """Pure function: voxels of \`\`kinds\`\` for box [start, stop) at \`\`level\`\`."""
    step = 2**level
    z = (np.arange(start[0], stop[0]) * step)[:, None, None].astype(F32)
    y = (np.arange(start[1], stop[1]) * step)[None, :, None].astype(F32)
    x = (np.arange(start[2], stop[2]) * step)[None, None, :].astype(F32)
    shape = tuple(b - a for a, b in zip(start, stop))
    out = np.zeros(shape, dtype=F32)
    for kind in kinds:
        if kind == "blobs" or kind == "shells":
            out += _spheres(z, y, x, seed, kind)
        elif kind == "noise":
            out += F32(80) * _fbm(z, y, x, seed)
        elif kind == "julia":
            out += _julia(z, y, x, seed, scale=max(full_shape) / 2.8)
        elif kind == "mandelbulb":
            out += _mandelbulb(start, stop, step, full_shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class SyntheticSource(Source):
    def __init__(
        self,
        kinds: list[str],
        shape,
        chunk_shape,
        level: int,
        seed: int,
        voxel_size: float,
        unit: str,
    ):
        self.kinds = tuple(kinds)
        self.level = level
        self.seed = seed
        self.step = 2**level
        self._full = tuple(int(s) for s in shape)
        self._info = ArrayInfo(
            shape=tuple(-(-s // self.step) for s in self._full),
            dtype=np.uint8,
            chunk_shape=tuple(chunk_shape),
            voxel_size=(voxel_size * self.step,) * 3,
            units=(unit,) * 3,
            axes=("z", "y", "x"),
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return f"synthetic:{'+'.join(self.kinds)}:{self._full}:{self.seed}:s{self.level}"

    def read(self, box: Box) -> np.ndarray:
        return generate(
            self.kinds, self._full, self.level, self.seed, tuple(box.start), tuple(box.stop)
        )


def open_synthetic(url: str) -> MultiscaleSource:
    parts = urlsplit(url)
    kinds = [k for k in parts.netloc.split("+") if k]
    for k in kinds:
        if k not in _KINDS:
            raise ValueError(f"unknown synthetic kind {k!r}; choose from {_KINDS}")
    q = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    shape = tuple(int(s) for s in q.get("shape", "1024,1024,1024").split(","))
    chunk = tuple(int(s) for s in q.get("chunk", "64,64,64").split(","))
    levels = int(q.get("levels", "0")) or max(1, int(np.ceil(np.log2(max(shape) / 256))) + 1)
    seed = int(q.get("seed", "0"))
    voxel_size = float(q.get("voxel_size", "8"))
    unit = q.get("unit", "nm")
    if len(shape) != 3:
        raise ValueError("synthetic sources are 3-D: shape=z,y,x")
    src = [
        SyntheticSource(kinds or ["blobs"], shape, chunk, lvl, seed, voxel_size, unit)
        for lvl in range(levels)
    ]
    return MultiscaleSource(src, name=url)
`,f=`"""Meshes computed when a viewer asks for them, in Neuroglancer's precomputed (legacy) mesh
format: one segment, \`\`1\`\`, whose fragments are the chunks of one level of a pipeline, each
meshed when it is fetched (the format names fragments in a manifest and leaves them to be
fetched one by one, so nothing is computed up front; its multi-resolution sibling needs
every fragment's byte offsets first).

    <dataset>/mesh/info        {"@type": "neuroglancer_legacy_mesh"}
    <dataset>/mesh/1:0         {"fragments": ["1:0:0_0_0", ...]}, every chunk of the level
    <dataset>/mesh/1:0:i_j_k   chunk (i, j, k) meshed: uint32 vertex count, float32 x, y, z
                               per vertex (nanometres), uint32 triangle corners

With \`\`lods\`\` above 1, a surface is served in Neuroglancer's multi-resolution format instead,
whose meshes get finer where the viewer zooms in:

    <dataset>/mesh/info        {"@type": "neuroglancer_multilod_draco", ...}
    <dataset>/mesh/1.index     the octree: per level of detail, the nodes and their sizes
    <dataset>/mesh/1           every node's fragment, one after another (HTTP Range requests)

Level of detail \`\`i\`\` is pyramid level \`\`level - lods + 1 + i\`\`, its nodes the chunks of that
level, so the coarsest is \`\`level\`\`'s chunks and each finer one splits them in eight. The
format lists every fragment's byte size before any is fetched, so each is padded to
\`\`FRAGMENT_BYTES\`\` (Draco decoders read what they need and ignore the rest) and its mesh
kept under \`\`MAX_TRIANGLES\`\`; nodes are listed only near the surface of the coarsest level
(\`\`multires_nodes\`\`), so the index stays small while the finer levels exist only where they
are fetched.

Two kinds. \`\`surface\`\`: the boundary of the voxels at or above \`\`threshold\`\`, by marching
cubes (scikit-image), from the chunk and one voxel more on its high sides so neighbouring
fragments meet, closed at the array's edges. \`\`terrain\`\`: an elevation model (\`\`y, x\`\`, or
\`\`z, y, x\`\` with one z) as a surface of two triangles per cell, the elevation times
\`\`exaggeration\`\` as height; cells with a NaN corner are left out. Imports nothing but
numpy, pydantic and scikit-image, so the browser engine's workers run it too.
"""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from chunkmirage.core import ArrayInfo, Box

_TO_NM = {"": 1.0, "nm": 1.0, "nanometer": 1.0, "um": 1e3, "micrometer": 1e3, "µm": 1e3,
          "mm": 1e6, "millimeter": 1e6, "m": 1e9, "meter": 1e9, "km": 1e12, "kilometer": 1e12}  # fmt: skip
MAX_SIDE = 512  # the default level: the finest whose longest side is at most this
FRAGMENT_BYTES = 1 << 17  # every multi-resolution fragment, padded: sizes are listed up front
MAX_TRIANGLES = 100_000  # a fragment's mesh, coarsened above it: Draco takes under a byte each
BITS = 16  # multi-resolution vertex positions: integers in [0, 2**BITS) across a node
BAND = 2  # nodes are listed within this many coarsest-level voxels of its surface
MAX_NODES = 2_000_000  # an index's nodes (16 bytes each): fewer levels of detail past it


class MeshSpec(BaseModel):
    """What a dataset's \`\`mesh\`\` frontend meshes."""

    kind: Literal["surface", "terrain"] = Field(
        "surface", description="surface: an isosurface of the volume; terrain: an elevation model"
    )
    level: int | None = Field(
        None,
        ge=0,
        description=f"The level meshed; default the finest whose longest side is at most {MAX_SIDE}",
    )
    threshold: float = Field(128.0, description="surface: values at or above this are inside")
    exaggeration: float = Field(1.0, gt=0, description="terrain: the elevation's scale")
    lods: int = Field(
        1,
        ge=1,
        le=8,
        description="surface: levels of detail. 1 is one mesh of \`level\`; more serve "
        "Neuroglancer's multi-resolution format, \`level\` the coarsest and each further one a "
        "pyramid level finer, meshed where the viewer zooms in",
    )


def mesh_level(infos: list[ArrayInfo], spec: MeshSpec) -> int:
    if spec.level is not None:
        if spec.level >= len(infos):
            raise ValueError(f"mesh level {spec.level}: the dataset has {len(infos)} levels")
        return spec.level
    small = [i for i, info in enumerate(infos) if max(info.shape[-3:]) <= MAX_SIDE]
    return small[0] if small else len(infos) - 1


def fragment_names(info: ArrayInfo) -> list[str]:
    return ["1:0:" + "_".join(map(str, idx)) for idx in np.ndindex(*info.chunk_grid)]


def fragment_box(info: ArrayInfo, index) -> Box:
    """A fragment's voxels: its chunk and one more on each high side (clipped), so its
    surface meets its neighbours'."""
    box = info.chunk_box(index)
    return Box(box.start, tuple(min(b + 1, s) for b, s in zip(box.stop, info.shape))).clip(
        info.shape
    )


def _nm(info: ArrayInfo) -> np.ndarray:
    return np.array([v * _TO_NM.get(u, 1.0) for v, u in zip(info.voxel_size, info.units)])


def fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo) -> bytes:
    """Mesh \`\`block\`\` (the level's voxels over \`\`box\`\`), vertices in nanometres."""
    scale = _nm(info)
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(info.translation, info.units)])
    if spec.kind == "terrain":
        return _terrain(spec, block, box, info, scale, origin)
    mask = np.asarray(block) >= spec.threshold
    if mask.ndim != 3:
        raise ValueError(f"a surface mesh needs a z, y, x volume, not {mask.ndim} axes")
    # closed at the array's edges: one voxel of outside beyond them
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))
    if not padded.any() or padded.all():
        return encode(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), 0.5)
    zyx = (verts - lo + np.array(box.start)) * scale[-3:] + origin[-3:]
    return encode(zyx[:, ::-1].astype(np.float32), faces[:, ::-1].astype(np.uint32))


def _terrain(spec, block, box: Box, info: ArrayInfo, scale, origin) -> bytes:
    z = np.asarray(block, dtype=np.float64)
    z = z.reshape(z.shape[-2:])  # y, x (a z of one dropped)
    h, w = z.shape
    y0, x0 = box.start[-2:]
    ys = (np.arange(y0, y0 + h) * scale[-2] + origin[-2])[:, None]
    xs = (np.arange(x0, x0 + w) * scale[-1] + origin[-1])[None, :]
    up = _TO_NM.get(info.units[-1], 1.0) * spec.exaggeration  # elevation in the grid's unit
    verts = np.stack(np.broadcast_arrays(xs, ys, np.nan_to_num(z) * up), -1).reshape(-1, 3)
    i = np.arange(h * w).reshape(h, w)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    ok = ~(
        np.isnan(z[:-1, :-1]) | np.isnan(z[:-1, 1:]) | np.isnan(z[1:, :-1]) | np.isnan(z[1:, 1:])
    )
    faces = np.concatenate([np.stack([a, c, b], -1)[ok], np.stack([b, c, d], -1)[ok]])
    return encode(verts.astype(np.float32), faces.astype(np.uint32))


def encode(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A legacy mesh fragment: vertex count, the vertices, the triangles' corners (LE)."""
    return (
        struct.pack("<I", len(verts))
        + np.ascontiguousarray(verts, "<f4").tobytes()
        + np.ascontiguousarray(faces, "<u4").tobytes()
    )


# ------------------------------------------------------------------ multi-resolution
def lod_levels(infos: list[ArrayInfo], spec: MeshSpec) -> list[int]:
    """The pyramid level of each level of detail, finest first."""
    if spec.kind != "surface" and spec.lods > 1:
        raise ValueError("levels of detail are for surface meshes")
    top = mesh_level(infos, spec)
    if top - spec.lods + 1 < 0:
        raise ValueError(f"lods={spec.lods} from level {top}: the dataset has levels 0..{top} below it")
    return list(range(top - spec.lods + 1, top + 1))


def _morton(xyz: np.ndarray) -> np.ndarray:
    """Z-curve keys of (n, 3) x, y, z positions (x the lowest bit)."""
    key = np.zeros(len(xyz), np.uint64)
    v = xyz.astype(np.uint64)
    for b in range(21):
        for a in range(3):
            key |= ((v[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return key


def surface_band(mask: np.ndarray, core: Box | None = None) -> np.ndarray:
    """The voxels within \`\`BAND\`\` of the surface of \`\`mask\`\` (inside or not; beyond its edges
    is outside), cropped to \`\`core\`\` (where \`\`mask\`\` is a block with a \`\`BAND\`\` border, so the
    parts of a level can be done apart and joined)."""
    from scipy.ndimage import maximum_filter, minimum_filter

    m = np.asarray(mask).astype(np.uint8)
    size = 2 * BAND + 1  # separable passes: a cube's dilation and erosion
    band = maximum_filter(m, size, mode="constant", cval=0) > minimum_filter(m, size, mode="constant", cval=0)
    return band if core is None else band[core.slices()]


def multires_nodes(spec: MeshSpec, band: np.ndarray, infos: list[ArrayInfo], chunk) -> list[np.ndarray]:
    """Per level of detail (finest first), its nodes, (n, 3) chunk positions \`\`z, y, x\`\` of its
    level in Z-curve order: those that meet \`\`band\`\`, the voxels of the coarsest level near its
    surface (\`\`surface_band\`\`). A finer level's surface lies near the coarser one's, and a node
    listed but empty only costs its fetch."""
    levels = lod_levels(infos, spec)
    band = np.asarray(band, bool)
    chunk = np.asarray(chunk)
    out = []
    for i, level in enumerate(levels):
        scale = 2 ** (len(levels) - 1 - i)  # this level's voxels per coarsest voxel
        grid = -(-np.asarray(infos[level].shape[-3:]) // chunk)
        if (chunk % scale == 0).all():  # a node is a block of whole coarsest voxels
            b = chunk // scale
            pad = np.zeros(tuple(grid * b), bool)
            pad[tuple(slice(0, min(n, g)) for n, g in zip(band.shape, grid * b))] = band[tuple(slice(0, g) for g in grid * b)]
            nodes = np.argwhere(pad.reshape(grid[0], b[0], grid[1], b[1], grid[2], b[2]).any((1, 3, 5)))
        else:  # nodes smaller than a coarsest voxel: every node each band voxel covers
            f = scale // chunk
            at = np.argwhere(band)
            offsets = np.array(np.meshgrid(*[np.arange(n) for n in f], indexing="ij")).reshape(3, -1).T
            nodes = (at[:, None, :] * f + offsets[None]).reshape(-1, 3)
            nodes = nodes[(nodes < grid).all(1)]
        out.append(nodes[np.argsort(_morton(nodes[:, ::-1]), kind="stable")])
    if (n := sum(len(o) for o in out)) > MAX_NODES:
        raise ValueError(f"lods={spec.lods} lists {n} mesh nodes, over {MAX_NODES}: ask for fewer levels of detail")
    return out


def multires_index(nodes: list[np.ndarray], infos: list[ArrayInfo], levels: list[int], chunk) -> bytes:
    """The \`\`1.index\`\` manifest: nodes sized \`\`chunk\`\` voxels of each level (nanometres, x y z),
    each level's origin offset from the finest's, every fragment \`\`FRAGMENT_BYTES\`\`."""
    first = infos[levels[0]]
    voxel = _nm(first)[-3:]
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(first.translation, first.units)])[-3:]
    shape = np.asarray(chunk, float) * voxel
    lod_scales = [float(_nm(infos[lv])[-3:].min()) for lv in levels]
    offsets = []
    for lv in levels:
        i = infos[lv]
        o = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(i.translation, i.units)])[-3:]
        offsets.extend((o - origin)[::-1])
    out = struct.pack("<3f", *shape[::-1]) + struct.pack("<3f", *origin[::-1]) + struct.pack("<I", len(levels))
    out += struct.pack(f"<{len(levels)}f", *lod_scales) + struct.pack(f"<{3 * len(levels)}f", *offsets)
    out += struct.pack(f"<{len(levels)}I", *[len(n) for n in nodes])
    for n in nodes:
        out += np.ascontiguousarray(n[:, ::-1].T, "<u4").tobytes() + np.full(len(n), FRAGMENT_BYTES, "<u4").tobytes()
    return out


def multires_info() -> dict:
    return {"@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": BITS,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "lod_scale_multiplier": 1}


def node_box(info: ArrayInfo, node, chunk) -> Box:
    """A node's voxels: its chunk of the level and one more on its high sides (clipped)."""
    start = tuple(int(n) * c for n, c in zip(node, chunk))
    stop = tuple(min(s + c + 1, n) for s, c, n in zip(start, chunk, info.shape))
    return Box(start, stop)


def multires_fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo, chunk) -> tuple[np.ndarray, np.ndarray]:
    """A node's mesh as the format stores it: vertices as integers across the node (x, y, z,
    \`\`[0, 2**BITS)\`\`) and triangles, marching cubes run on each octant apart so no triangle
    crosses one (as coarser levels must be), coarsened past \`\`MAX_TRIANGLES\`\`."""
    from skimage.measure import marching_cubes

    mask = np.asarray(block) >= spec.threshold
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))  # closed at the array's edges
    chunk = np.asarray(chunk)
    half = chunk // 2
    for step in (1, 2, 4, 8):
        verts, faces, n = [], [], 0
        for octant in np.ndindex(2, 2, 2):
            o = np.array(octant) * half
            stop = np.where(octant, chunk, half) + 1
            # in the padded block, node voxel j is at j + lo
            sub = padded[tuple(slice(a + e, b + e) for a, b, e in zip(o, stop, lo))]
            if min(sub.shape) < 2 or sub.all() or not sub.any():
                continue
            v, f, _, _ = marching_cubes(sub.astype(np.float32), 0.5, step_size=step)
            verts.append(v + o)
            faces.append(f + n)
            n += len(v)
        if not verts:
            return np.zeros((0, 3), np.uint32), np.zeros((0, 3), np.uint32)
        f = np.concatenate(faces)
        if len(f) <= MAX_TRIANGLES or step == 8:
            break
    v = np.concatenate(verts)  # node coordinates: the node's voxel 0 at 0
    q = np.clip(np.rint(v / chunk * (2**BITS - 1)), 0, 2**BITS - 1).astype(np.uint32)
    return np.ascontiguousarray(q[:, ::-1]), np.ascontiguousarray(f[:, ::-1].astype(np.uint32))


def encode_draco(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A fragment as Neuroglancer reads it: Draco, its integer positions kept as they are
    (quantized to \`\`BITS\`\` over exactly \`\`[0, 2**BITS - 1]\`\`), padded to \`\`FRAGMENT_BYTES\`\`.
    Needs DracoPy (the \`\`mesh\`\` extra); the browser engine encodes with Draco's own wasm."""
    import DracoPy

    if not len(faces):
        return b"\\0" * FRAGMENT_BYTES
    data = DracoPy.encode(verts, faces, quantization_bits=BITS, quantization_range=2**BITS - 1,
                          quantization_origin=[0, 0, 0], compression_level=7)
    if len(data) > FRAGMENT_BYTES:
        raise ValueError(f"a mesh fragment took {len(data)} bytes, over {FRAGMENT_BYTES}")
    return data + b"\\0" * (FRAGMENT_BYTES - len(data))
`,p=`"""Stitching overlapping tiles by interest points, as BigStitcher registers them: blobs found
in each overlap by a difference of Gaussians, matched between neighbouring tiles by the
constellation of their nearest neighbours, the matches filtered by RANSAC, and every tile's
placement fitted to all the kept matches at once. Then the tiles are fused, blended where
they overlap, region by region: nothing is written.

Coordinates are physical, C order \`\`(z, y, x)\`\`. A tile is placed by an affine from its
voxels to the scene (rows \`\`[A | t]\`\`): its stage position (where the microscope put it),
which stitching corrects by a model of its own (translation, rigid or affine) on top.

Each step is a function of arrays and JSON-able values, so the browser engine's workers run
them as they are (numpy and scipy only), and \`\`stitch://\`\` (\`\`sources.stitch\`\`) runs them
when it is opened. \`\`tiles_from_bdv\`\` reads the tiles and their placement from a BigStitcher
(BigDataViewer) project, the XML its tools write.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MIN_MATCHES = {"translation": 1, "rigid": 3, "affine": 4}  # a model's smallest sample
BATCH = 512  # RANSAC hypotheses scored at once


class StitchParams(BaseModel):
    """How tiles are stitched: the interest points, their matching, RANSAC and the fit."""

    channel: int = Field(0, ge=0, description="The tiles' channel (setup attribute) stitched")
    level: int = Field(1, ge=0, description="The tiles' level interest points are found on")
    sigma: float = Field(
        1.8, gt=0, description="Blob size: the difference of Gaussians' smaller sigma, in "
        "voxels of that level along x (scaled along the other axes by their spacing)",
    )
    threshold: float = Field(
        0.005, gt=0, description="Smallest difference-of-Gaussians peak kept, in units of the "
        "tile's intensity range (BigStitcher's threshold)",
    )
    margin: float = Field(
        20.0, ge=0, description="How far tiles may be from their stage positions: the "
        "overlaps searched are grown by this (physical units)",
    )
    neighbors: int = Field(3, ge=2, le=6, description="Nearest neighbours a point's descriptor holds")
    redundancy: int = Field(1, ge=0, le=3, description="Extra neighbours: subsets of them are tried too")
    significance: float = Field(
        3.0, ge=1, description="A match's descriptor must be this many times closer than the "
        "next best candidate's",
    )
    model: Literal["translation", "rigid", "affine"] = Field(
        "translation", description="What each tile may do beyond its stage position"
    )
    epsilon: float = Field(5.0, gt=0, description="RANSAC: largest error of an inlier (physical units)")
    min_inlier_ratio: float = Field(0.1, ge=0, le=1, description="RANSAC: smallest share of inliers")
    min_inliers: int = Field(6, ge=1, description="RANSAC: fewest inliers for a pair to count")
    iterations: int = Field(10000, ge=1, le=200000, description="RANSAC: hypotheses tried")
    seed: int = Field(0, description="RANSAC's random draws")
    blend: float = Field(
        40.0, ge=0, description="Fusion: the band at a tile's edges its weight falls off over "
        "(physical units along y and x)",
    )

    @classmethod
    def from_query(cls, q: dict[str, str]) -> StitchParams:
        if unknown := set(q) - set(cls.model_fields):
            raise ValueError(
                f"unknown stitch:// parameters {sorted(unknown)}; allowed: {sorted(cls.model_fields)}"
            )
        return cls(**q)


# ------------------------------------------------------------------ affines
def to4(a) -> np.ndarray:
    m = np.eye(4)
    m[:3] = np.asarray(a, float).reshape(3, 4)
    return m


def apply(a, p: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return p @ a[:3, :3].T + a[:3, 3]


def compose(*affines) -> np.ndarray:
    """\`\`compose(a, b)(p) == a(b(p))\`\`."""
    m = np.eye(4)
    for a in affines:
        m = m @ to4(a)
    return m[:3]


def invert(a) -> np.ndarray:
    return np.linalg.inv(to4(a))[:3]


# ------------------------------------------------------------------ interest points
def _gaussian(x: np.ndarray, sigma) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma, mode="nearest", truncate=3.0)


def detect(block: np.ndarray, voxel, sigma: float, threshold: float, lo: float, hi: float) -> np.ndarray:
    """Bright blobs in \`\`block\`\`: local maxima of a difference of Gaussians (sigma and
    2^(1/4) sigma, BigStitcher's), at least \`\`threshold\`\` of the intensity range [lo, hi],
    to subvoxel precision. Returns \`\`(n, 3)\`\` voxel positions in \`\`block\`\`; none within a
    voxel of its faces (their maxima are cut off)."""
    from scipy.ndimage import maximum_filter

    if min(block.shape) < 3:
        return np.zeros((0, 3))
    voxel = np.asarray(voxel, float)
    s = sigma * voxel[-1] / voxel  # the same physical size along every axis
    x = (np.asarray(block, np.float32) - lo) / max(hi - lo, 1e-12)
    dog = _gaussian(x, s) - _gaussian(x, s * 2 ** 0.25)
    peak = (dog == maximum_filter(dog, size=3, mode="nearest")) & (dog >= threshold)
    peak[[0, -1]] = peak[:, [0, -1]] = peak[:, :, [0, -1]] = False
    at = np.argwhere(peak)
    if not len(at):
        return np.zeros((0, 3))
    # a parabola through each peak and its two neighbours, per axis
    out = at.astype(float)
    v = dog[tuple(at.T)]
    for a in range(3):
        up, down = at.copy(), at.copy()
        up[:, a] += 1
        down[:, a] -= 1
        f1, f0 = dog[tuple(up.T)], dog[tuple(down.T)]
        curve = f1 + f0 - 2 * v
        out[:, a] += np.where(curve < 0, 0.5 * (f0 - f1) / np.where(curve < 0, curve, -1), 0).clip(-0.5, 0.5)
    return out


def points_in(tile: dict, level: int, block: np.ndarray, start, voxel, lo, hi, p: StitchParams) -> np.ndarray:
    """A tile's interest points in the scene box \`\`[lo, hi]\`\`, from \`\`block\`\` (its level's
    voxels from \`\`start\`\`, as \`\`region\`\` gives them): scene positions at its stage placement."""
    block = np.asarray(block)
    v = detect(block, voxel, p.sigma, p.threshold, float(block.min()), float(block.max()))
    w = apply(compose(tile["stage"], level_to_base(tile, level)), v + np.asarray(start))
    return w[((w >= lo) & (w <= hi)).all(1)]


# ------------------------------------------------------------------ matching
def descriptors(points: np.ndarray, neighbors: int, redundancy: int):
    """Each point's constellation: the offsets to \`\`neighbors\`\` of its nearest
    \`\`neighbors + redundancy\`\` (every such subset, nearest first), concatenated. Unchanged
    by a translation, so tiles placed only roughly still match. Returns the descriptors and
    the point each belongs to."""
    from scipy.spatial import cKDTree

    k = neighbors + redundancy
    if len(points) <= k:
        return np.zeros((0, 3 * neighbors)), np.zeros(0, int)
    _, near = cKDTree(points).query(points, k + 1)
    offsets = points[near[:, 1:]] - points[:, None, :]  # (n, k, 3), nearest first
    subsets = list(itertools.combinations(range(k), neighbors))
    d = np.concatenate([offsets[:, list(s)].reshape(len(points), -1) for s in subsets])
    owner = np.tile(np.arange(len(points)), len(subsets))
    return d, owner


def match(a: np.ndarray, b: np.ndarray, neighbors: int = 3, redundancy: int = 1, significance: float = 3.0) -> np.ndarray:
    """Candidate matches \`\`(i, j)\`\` of points \`\`a[i]\`\` and \`\`b[j]\`\`: \`\`b[j]\`\` is the point
    whose descriptors come closest to one of \`\`a[i]\`\`'s, \`\`significance\`\` times closer than
    any other point's, and the other way round."""
    from scipy.spatial import cKDTree

    da, oa = descriptors(a, neighbors, redundancy)
    db, ob = descriptors(b, neighbors, redundancy)
    if not len(da) or not len(db):
        return np.zeros((0, 2), int)

    def best(dx, ox, dy, oy, nx):
        """For each point of x: its nearest point of y, and how much nearer than the next."""
        k = min(len(dy), 8)
        dist, at = cKDTree(dy).query(dx, k)
        dist, at = dist.reshape(len(dx), k), at.reshape(len(dx), k)
        rows = np.repeat(ox, k)
        cols = oy[at.ravel()]
        dist = dist.ravel()
        # each (x point, y point): the closest of their descriptors
        order = np.lexsort((dist, cols, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        first = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
        rows, cols, dist = rows[first], cols[first], dist[first]
        order = np.lexsort((dist, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        start = np.r_[True, rows[1:] != rows[:-1]]
        idx = np.flatnonzero(start)
        nearest = np.full(nx, -1)
        ratio = np.zeros(nx)
        nearest[rows[idx]] = cols[idx]
        nxt = idx + 1
        has = (nxt < len(rows)) & (np.r_[rows, -1][nxt] == rows[idx])
        second = np.where(has, np.r_[dist, np.inf][np.where(has, nxt, len(dist))], np.inf)
        ratio[rows[idx]] = second / np.maximum(dist[idx], 1e-12)
        return nearest, ratio

    ab, ra = best(da, oa, db, ob, len(a))
    ba, _ = best(db, ob, da, oa, len(b))
    i = np.flatnonzero((ab >= 0) & (ra >= significance))
    i = i[ba[ab[i]] == i]  # mutual
    return np.stack([i, ab[i]], 1)


# ------------------------------------------------------------------ models
def fit(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The \`\`model\`\` taking points \`\`p\`\` nearest to \`\`q\`\` (least squares), as \`\`[A | t]\`\`."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if model == "translation":
        return np.hstack([np.eye(3), (q - p).mean(0)[:, None]])
    cp, cq = p.mean(0), q.mean(0)
    if model == "rigid":
        u, _, vt = np.linalg.svd((q - cq).T @ (p - cp))
        r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        return np.hstack([r, (cq - r @ cp)[:, None]])
    x = np.hstack([p - cp, np.ones((len(p), 1))])
    sol = np.linalg.lstsq(x, q - cq, rcond=None)[0]  # (4, 3)
    a = sol[:3].T
    return np.hstack([a, (cq + sol[3] - a @ cp)[:, None]])


def _fit_many(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """\`\`fit\`\` of each of a batch of minimal samples \`\`(m, k, 3)\`\`: \`\`(m, 3, 4)\`\`."""
    cp, cq = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    m = len(p)
    if model == "translation":
        out = np.zeros((m, 3, 4))
        out[:, :, :3] = np.eye(3)
        out[:, :, 3] = (cq - cp)[:, 0]
        return out
    if model == "rigid":
        u, _, vt = np.linalg.svd(np.einsum("mki,mkj->mij", q - cq, p - cp))
        d = np.sign(np.linalg.det(u @ vt))
        u[:, :, 2] *= d[:, None]
        a = u @ vt
    else:
        x = np.concatenate([p - cp, np.ones(p.shape[:2] + (1,))], 2)  # (m, 4, 4)
        x = x + np.eye(4)[None] * 1e-9  # degenerate samples stay finite (and score badly)
        sol = np.linalg.solve(x, q - cq)  # (m, 4, 3)
        a = np.transpose(sol[:, :3], (0, 2, 1))
        cq = cq + sol[:, 3:4]
    t = cq[:, 0] - np.einsum("mij,mj->mi", a, cp[:, 0])
    return np.concatenate([a, t[:, :, None]], 2)


def ransac(p: np.ndarray, q: np.ndarray, model: str, epsilon: float, min_inlier_ratio: float,
           min_inliers: int, iterations: int, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """The \`\`model\`\` that most candidate matches \`\`p[i] -> q[i]\`\` agree with to within
    \`\`epsilon\`\`, from \`\`iterations\`\` random minimal samples, refitted to its inliers until
    they stop changing. Returns the model (\`\`None\`\` if too few agree) and the inliers."""
    n, k = len(p), MIN_MATCHES[model]
    none = np.zeros(n, bool)
    if n < max(k, min_inliers):
        return None, none
    rng = np.random.default_rng(seed)
    best, best_count = none, 0
    for start in range(0, iterations, BATCH):
        m = min(BATCH, iterations - start)
        pick = np.argsort(rng.random((m, n)), 1)[:, :k] if k > 1 else rng.integers(0, n, (m, 1))
        models = _fit_many(model, p[pick], q[pick])
        d = p @ models[:, :, :3].transpose(0, 2, 1) + models[:, None, :, 3] - q  # (m, n, 3)
        err = np.sqrt((d * d).sum(2))
        counts = (err < epsilon).sum(1)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), err[i] < epsilon
    for _ in range(20):  # refit to the inliers until they settle
        if best.sum() < k:
            return None, none
        a = fit(model, p[best], q[best])
        now = np.linalg.norm(apply(a, p) - q, axis=1) < epsilon
        if (now == best).all():
            break
        best = now
    if best.sum() < max(k, min_inliers) or best.mean() < min_inlier_ratio:
        return None, best
    return fit(model, p[best], q[best]), best


def optimize(n: int, links: list[tuple[int, int, np.ndarray, np.ndarray]], model: str,
             fixed: int = 0, rounds: int = 2000, tolerance: float = 1e-4) -> tuple[list[np.ndarray], list[bool]]:
    """Every tile's correction (applied after its stage placement) fitted to all the kept
    matches at once: link \`\`(i, j, p, q)\`\` asks tile \`\`i\`\`'s points \`\`p\`\` to meet tile
    \`\`j\`\`'s \`\`q\`\` (both placed by their stage positions). Each round refits each tile to
    where its neighbours put their ends of its matches. Of each group of tiles that links
    join, one stays where it is: \`\`fixed\`\` in its group, the first tile in the others.
    Returns the corrections and which tiles were joined to another."""
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    models = [identity.copy() for _ in range(n)]
    group = list(range(n))

    def find(t):
        while group[t] != t:
            t = group[t]
        return t

    for i, j, _, _ in links:
        group[find(i)] = find(j)
    anchors = {}
    for t in [fixed, *range(n)]:
        anchors.setdefault(find(t), t)
    joined = [any(t in (i, j) for i, j, _, _ in links) for t in range(n)]
    order = [t for t in range(n) if joined[t] and t not in anchors.values()]
    for _ in range(rounds):
        moved = 0.0
        for t in order:
            p, q = [], []
            for i, j, pi, qj in links:
                if i == t:
                    p.append(pi)
                    q.append(apply(models[j], qj))
                elif j == t:
                    p.append(qj)
                    q.append(apply(models[i], pi))
            if not p:
                continue
            p, q = np.concatenate(p), np.concatenate(q)
            new = fit(model, p, q)
            moved = max(moved, float(np.abs(apply(new, p) - apply(models[t], p)).max()))
            models[t] = new
        if moved < tolerance:
            break
    return models, joined


# ------------------------------------------------------------------ the whole registration
def overlaps(tiles: list[dict], margin: float) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Pairs of tiles whose stage placements overlap, and the overlap's scene box (grown by
    \`\`margin\`\`), as \`\`(i, j, lo, hi)\`\`."""
    boxes = [bounds(t["stage"], t["shape"][0]) for t in tiles]
    out = []
    for i, j in itertools.combinations(range(len(tiles)), 2):
        lo = np.maximum(boxes[i][0], boxes[j][0])
        hi = np.minimum(boxes[i][1], boxes[j][1])
        if (hi > lo).all():
            out.append((i, j, lo - margin, hi + margin))
    return out


def bounds(affine, shape) -> tuple[np.ndarray, np.ndarray]:
    """The scene box an affine puts a volume of \`\`shape\`\` voxels in (voxel centres at
    integers, so a voxel's extent is +-0.5)."""
    corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in shape])))
    w = apply(affine, corners)
    return w.min(0), w.max(0)


def region(tile: dict, level: int, lo, hi, placement=None) -> tuple[list[int], list[int]] | None:
    """The voxels \`\`[start, stop)\`\` of a tile's level holding the scene box \`\`[lo, hi]\`\` (a
    voxel more on each side, for interpolation) where \`\`placement\`\` (default: its stage
    position) puts it, or \`\`None\`\` if it holds none of it."""
    to_scene = compose(tile["stage"] if placement is None else placement, level_to_base(tile, level))
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    v = apply(invert(to_scene), corners)
    start = np.maximum(np.floor(v.min(0)) - 1, 0).astype(int)
    stop = np.minimum(np.ceil(v.max(0)) + 2, tile["shape"][level]).astype(int)
    if (stop <= start).any():
        return None
    return start.tolist(), stop.tolist()


def level_to_base(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxels to its level-0 voxels: each level halves (or so) its shape;
    its voxel centres sit at the centres of the level-0 voxels they cover."""
    f = np.array(tile["shape"][0], float) / np.array(tile["shape"][level], float)
    f = np.round(f) if np.allclose(f, np.round(f), atol=0.05) else f
    return np.hstack([np.diag(f), ((f - 1) / 2)[:, None]])


def register(tiles: list[dict], points: list[tuple[np.ndarray, np.ndarray]], p: StitchParams,
             fixed: int = 0) -> dict:
    """Matches, RANSAC and the global fit from each overlapping pair's interest points
    (\`\`points[k]\`\`: pair \`\`k\`\` of \`\`overlaps\`\`, both tiles' points in scene coordinates at
    their stage placement). JSON-able: per pair the candidates, inliers and errors; per
    tile its correction and its final placement."""
    pairs = overlaps(tiles, p.margin)
    links, report = [], []
    for k, (i, j, _, _) in enumerate(pairs):
        a, b = (np.asarray(x, float).reshape(-1, 3) for x in points[k])
        m = match(a, b, p.neighbors, p.redundancy, p.significance)
        model, inliers = ransac(a[m[:, 0]], b[m[:, 1]], p.model, p.epsilon, p.min_inlier_ratio,
                                p.min_inliers, p.iterations, p.seed)
        kept = m[inliers] if model is not None else m[:0]
        if model is not None:
            links.append((i, j, a[kept[:, 0]], b[kept[:, 1]]))
        report.append({
            "tiles": [i, j], "points": [len(a), len(b)], "candidates": len(m),
            "inliers": int(len(kept)), "kept": model is not None,
            # the matches, scene positions of both ends at the stage placement
            "a": a[m[:, 0]].round(3).tolist(), "b": b[m[:, 1]].round(3).tolist(),
            "inlier": inliers.tolist(),
        })
    corrections, placed = optimize(len(tiles), links, p.model, fixed)
    for r in report:
        i, j = r["tiles"]
        if r["kept"]:
            sel = np.array(r["inlier"], bool)
            a, b = np.array(r["a"])[sel], np.array(r["b"])[sel]
            e = np.linalg.norm(apply(corrections[i], a) - apply(corrections[j], b), axis=1)
            r["error"] = {"mean": float(e.mean()), "max": float(e.max())}
    return {
        "pairs": report,
        "corrections": [c.tolist() for c in corrections],
        "placed": placed,
        "placements": [compose(c, t["stage"]).tolist() for c, t in zip(corrections, tiles)],
    }


def compare(tiles: list[dict], placements: list) -> float | None:
    """How far the found placements are from the tiles' reference ones (a registration done
    before), as the root-mean-square distance of the tiles' centres once the two are
    lined up on average (stitching fixes the scene only up to a common shift)."""
    if not all(t.get("reference") is not None for t in tiles):
        return None
    c = [np.array(t["shape"][0], float)[None] / 2 - 0.5 for t in tiles]
    ours = np.concatenate([apply(a, x) for a, x in zip(placements, c)])
    theirs = np.concatenate([apply(t["reference"], x) for t, x in zip(tiles, c)])
    d = ours - theirs
    return float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))


# ------------------------------------------------------------------ fusion
def level_voxel(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxel size in the scene, at its stage placement."""
    return np.linalg.norm(compose(tile["stage"], level_to_base(tile, level))[:, :3], axis=0)


def grids(tiles: list[dict], placements: list) -> list[dict]:
    """The fused volume's levels, one per level every tile has: the first tile's voxel size
    there, and the origin and shape covering every tile where \`\`placements\`\` put them."""
    lo, hi = zip(*(bounds(a, t["shape"][0]) for a, t in zip(placements, tiles)))
    lo, hi = np.min(lo, 0), np.max(hi, 0)
    out = []
    for k in range(min(len(t["shape"]) for t in tiles)):
        voxel = level_voxel(tiles[0], k)
        shape = np.ceil((hi - lo) / voxel - 1e-6).astype(int)
        out.append({"shape": shape.tolist(), "voxel": voxel.tolist(), "origin": (lo + voxel / 2).tolist()})
    return out


def scene_box(grid: dict, out_lo, out_hi) -> tuple[np.ndarray, np.ndarray]:
    """The scene box that voxels \`\`[out_lo, out_hi)\`\` of a fused level cover."""
    voxel, origin = np.asarray(grid["voxel"]), np.asarray(grid["origin"])
    return origin + (np.asarray(out_lo) - 0.5) * voxel, origin + (np.asarray(out_hi) - 0.5) * voxel


def _weight(v: np.ndarray, shape, band: np.ndarray) -> np.ndarray:
    """A tile's blending weight at its voxel positions \`\`v\`\` (n, 3): 1 inside, falling off as
    a half cosine over \`\`band\`\` voxels from each face, 0 outside."""
    w = np.ones(len(v))
    for a in range(3):
        inside = (v[:, a] >= -0.5) & (v[:, a] <= shape[a] - 0.5)
        w = np.where(inside, w, 0)
        if band[a] > 0:
            d = np.minimum(v[:, a] + 0.5, shape[a] - 0.5 - v[:, a])
            w = w * np.where(d < band[a], 0.5 - 0.5 * np.cos(np.pi * np.clip(d, 0, None) / band[a]), 1)
    return w


def fuse(tiles: list[dict], placements: list, level: int, grid: dict, out_lo, out_hi,
         blocks: list[tuple[np.ndarray, list[int]] | None], blend: float, dtype) -> np.ndarray:
    """Voxels \`\`[out_lo, out_hi)\`\` of the fused volume on \`\`grid\`\`: each tile sampled
    (trilinearly) where its placement puts it, from \`\`blocks[t]\`\` (its level's voxels from
    a start, or \`\`None\`\` where it has none of the region), averaged with weights that fall
    off near its edges, so the seams do not show."""
    from scipy.ndimage import map_coordinates

    out_lo, out_hi = np.asarray(out_lo), np.asarray(out_hi)
    shape = tuple(out_hi - out_lo)
    idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(out_lo, out_hi)], indexing="ij"), -1)
    w_scene = np.asarray(grid["origin"]) + idx.reshape(-1, 3) * np.asarray(grid["voxel"])
    total = np.zeros(len(w_scene))
    weight = np.zeros(len(w_scene))
    for t, got in enumerate(blocks):
        if got is None:
            continue
        block, start = got
        tile = tiles[t]
        to_scene = compose(placements[t], level_to_base(tile, level))
        v = apply(invert(to_scene), w_scene)
        vox = np.abs(np.linalg.det(np.asarray(to_scene)[:, :3])) ** (1 / 3)
        band = np.array([0.0, blend, blend]) / vox  # along y and x: tiles share their z range
        w = _weight(v, tile["shape"][level], band)
        hit = w > 0
        if not hit.any():
            continue
        local = (v[hit] - np.asarray(start)).T
        total[hit] += w[hit] * map_coordinates(np.asarray(block, np.float32), local, order=1, mode="nearest")
        weight[hit] += w[hit]
    out = np.where(weight > 0, total / np.maximum(weight, 1e-12), 0).reshape(shape)
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


# ------------------------------------------------------------------ BigStitcher projects
def _affine_xyz(text: str) -> np.ndarray:
    """A BigDataViewer affine (12 numbers, rows of x, y, z) in C order (z, y, x)."""
    m = np.array([float(v) for v in text.split()]).reshape(3, 4)
    a = np.zeros((3, 4))
    a[:, :3] = m[::-1, :3][:, ::-1]
    a[:, 3] = m[::-1, 3]
    return a


def tiles_from_bdv(xml: str, base: str, channel: int = 0, drop: str = "Stitching Transform") -> list[dict]:
    """The tiles of one channel of a BigStitcher project (its XML text; \`\`base\`\`: the URL
    of the folder it is in), each with its stage placement: every view transform but the
    outermost ones named \`\`drop\`\` (the stitching found before, kept as the reference). Reads
    projects whose images are OME-Zarr (BigStitcher-Spark's \`\`bdv.multimg.zarr\`\` loader),
    one group per setup and time point."""
    root = ET.fromstring(xml)
    loader = root.find("SequenceDescription/ImageLoader")
    if loader is None or loader.get("format") != "bdv.multimg.zarr":
        raise ValueError(f"only OME-Zarr BigStitcher projects are read (bdv.multimg.zarr), not {loader.get('format') if loader is not None else 'none'}")
    zarr = loader.findtext("zarr", "").strip()
    folder = zarr if "://" in zarr or zarr.startswith("/") else f"{base.rstrip('/')}/{zarr}"
    groups = {(g.get("setup"), g.get("tp")): g.get("path") for g in loader.iter("zgroup")}
    tp = min({tp for _, tp in groups}, key=int)
    regs = {r.get("setup"): r for r in root.iter("ViewRegistration") if r.get("timepoint") == tp}
    tiles = []
    for vs in root.iter("ViewSetup"):
        attrs = vs.find("attributes")
        if attrs is None or int(attrs.findtext("channel", "0")) != channel:
            continue
        sid = vs.findtext("id").strip()
        size = [int(v) for v in vs.findtext("size").split()][::-1]  # x y z -> z y x
        transforms = regs[sid].findall("ViewTransform")
        names = [t.findtext("Name", "") for t in transforms]
        affines = [_affine_xyz(t.findtext("affine")) for t in transforms]
        n = 0
        while n < len(names) and names[n] == drop:
            n += 1
        tiles.append({
            "name": f"tile {attrs.findtext('tile', sid)}", "setup": int(sid),
            "url": f"{folder}/{groups[(sid, tp)]}", "select": {"t": 0, "c": 0},
            "shape": [size], "stage": compose(*affines[n:]).tolist(),
            "reference": compose(*affines).tolist() if n else None,
        })
    if not tiles:
        raise ValueError(f"no tiles of channel {channel}")
    return tiles
`,m=`"""Following one object through a time series of label images, frame by frame, as the
frames are read: in each next frame it is the label that overlaps it most. Labels need not
keep their ids from frame to frame (a segmentation done frame by frame rarely does); an
object that moves less than its own size between frames is followed by its overlap.

Each frame reads only a box around the object (its last bounding box, grown by a margin),
so following one nucleus through a whole time-lapse reads a sliver of it. A nucleus that
divides first collapses (its envelope breaks down in mitosis, and the segmentation loses
it); new nuclei then appear near where it was, as labels nothing covered the frame before
(\`\`newborns\`\`), and are taken for its daughters and followed in turn, so a track becomes a
lineage. That is a guess from where and when they appear: the segmentation does not say
which nucleus a new one came from, and in a dense colony a neighbour's daughter can be
taken for this one's. \`\`step\`\` and
\`\`newborns\`\` are the work of one frame, numpy only, which the browser engine's workers run
as they are; \`\`follow\`\` and \`\`lineage\`\` loop them over a source for Python callers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

MIN_OVERLAP = 0.2  # of the object's voxels: less and it is lost (it left, or the labels failed)
DIVIDED = 0.65  # a volume this fraction of the frame before's, or less: a division
NEWBORN = 0.4  # a label no label of the frame before covers this share of: new (a daughter, say)
GAP = 24  # frames searched for daughters after a nucleus collapses into mitosis and is lost
WITHIN = 18.0  # physical units (µm) from the mother that daughters are looked for


def measure(block: np.ndarray, label: int, start, voxel) -> dict | None:
    """Label \`\`label\`\` in \`\`block\`\` (whose voxel 0 is \`\`start\`\` of the level): its volume
    (voxels times their size), centroid and bounding box (level voxels, \`\`[lo, hi)\`\`), and
    whether it touches the block's faces (the block may not hold all of it)."""
    at = np.argwhere(np.asarray(block) == label)
    if not len(at):
        return None
    lo, hi = at.min(0), at.max(0) + 1
    shape = np.asarray(block.shape)
    touches = bool(((lo == 0) & (np.asarray(start) > 0)).any() or (hi == shape).any())
    start = np.asarray(start)
    return {
        "label": int(label),
        "volume": float(len(at) * np.prod(voxel)),
        "centroid": (at.mean(0) + start).tolist(),
        "lo": (lo + start).tolist(),
        "hi": (hi + start).tolist(),
        "touches": touches,
    }


def step(prev: np.ndarray, label: int, nxt: np.ndarray, start, voxel) -> dict | None:
    """The object labelled \`\`label\`\` in \`\`prev\`\`, in the next frame \`\`nxt\`\` (the same box of
    the level, from \`\`start\`\`): the label it overlaps most there, measured, with the overlap
    (its share of the object's voxels). \`\`None\`\` if under \`\`MIN_OVERLAP\`\`."""
    mask = np.asarray(prev) == label
    if not mask.any():
        return None
    under = np.asarray(nxt)[mask]
    under = under[under > 0]
    if len(under) < MIN_OVERLAP * mask.sum():
        return None
    ids, counts = np.unique(under, return_counts=True)
    best = int(ids[np.argmax(counts)])
    found = measure(nxt, best, start, voxel)
    if found is not None:
        found["overlap"] = float(counts.max() / mask.sum())
    return found


def box_around(record: dict, shape, margin) -> tuple[list[int], list[int]]:
    """The box the next frame is read over: the object's bounding box grown by \`\`margin\`\`
    voxels, within the level."""
    lo = np.maximum(np.asarray(record["lo"]) - margin, 0)
    hi = np.minimum(np.asarray(record["hi"]) + margin, shape)
    return lo.tolist(), hi.tolist()


def follow(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
           label: int, frames: range, margin) -> Iterator[tuple[int, dict]]:
    """The object labelled \`\`label\`\` at frame \`\`t0\`\`, through \`\`frames\`\` (counting up or down
    from \`\`t0\`\`); \`\`read(t, lo, hi)\`\` gives a frame's labels over a box of the level. Yields
    \`\`(t, record)\`\` for \`\`t0\`\` and each frame it is found in, until it is lost; a record
    says \`\`divided\`\` when its volume fell to \`\`DIVIDED\`\` of the frame before's."""
    margin = np.asarray(margin)
    # the whole object at t0: start from a box around its voxels in the whole frame
    first = measure(read(t0, [0] * len(shape), list(shape)), label, [0] * len(shape), voxel)
    if first is None:
        raise ValueError(f"no label {label} at frame {t0}")
    yield t0, first
    record, t = first, t0
    for nt in frames:
        lo, hi = box_around(record, shape, margin)
        prev, nxt = read(t, lo, hi), read(nt, lo, hi)
        found = step(prev, record["label"], nxt, lo, voxel)
        if found is not None and found["touches"]:  # the box cut it: read all of it
            lo, hi = box_around(found, shape, 2 * margin)
            found = step(read(t, lo, hi), record["label"], read(nt, lo, hi), lo, voxel)
        if found is None:
            return
        found["divided"] = found["volume"] <= DIVIDED * record["volume"]
        yield nt, found
        record, t = found, nt


def newborns(prev: np.ndarray, nxt: np.ndarray, start, voxel, centre, within: float = WITHIN,
             min_volume: float = 0.0) -> list[dict]:
    """Labels of \`\`nxt\`\` that are new: covered less than \`\`NEWBORN\`\` by any one label of the
    frame before, \`\`prev\`\` (the same box, from \`\`start\`\`); within \`\`within\`\` (physical units)
    of \`\`centre\`\` (level voxels) and of \`\`min_volume\`\` or more. After a mitosis, which hides
    the nucleus for a few frames, these are likely its daughters (a guess: nothing says which
    nucleus a new one came from). Nearest first."""
    nxt, prev, voxel = np.asarray(nxt), np.asarray(prev), np.asarray(voxel)
    out = []
    for i in np.unique(nxt[nxt > 0]):
        mask = nxt == i
        under = prev[mask]
        under = under[under > 0]
        if len(under) and np.bincount(under).max() >= NEWBORN * mask.sum():
            continue
        found = measure(nxt, int(i), start, voxel)
        far = np.linalg.norm((np.asarray(found["centroid"]) - np.asarray(centre)) * voxel)
        if far <= within and found["volume"] >= min_volume:
            found["distance"] = float(far)
            out.append(found)
    return sorted(out, key=lambda f: f["distance"])


def mother_of(branch: dict[int, dict], lost: int) -> dict | None:
    """If a branch lost at frame \`\`lost\`\` had collapsed first (its last volume \`\`DIVIDED\`\` of
    its largest in the dozen frames before, or less), the record at that largest: the
    nucleus that went into mitosis. \`\`None\`\` if it was just lost."""
    recent = [branch[t] for t in sorted(branch) if lost - 12 <= t < lost]
    if not recent:
        return None
    biggest = max(recent, key=lambda r: r["volume"])
    return biggest if recent[-1]["volume"] <= DIVIDED * biggest["volume"] else None


def daughters(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, mother: dict,
              lost: int, frames: int, gap: int = GAP, within: float = WITHIN) -> list[tuple[int, dict]]:
    """Up to two likely daughters of \`\`mother\`\` (a record, see \`\`mother_of\`\`): the first new
    nuclei (\`\`newborns\`\`) near it, looked for from frame
    \`\`lost\`\` for \`\`gap\`\` frames in a box \`\`within\`\` around it: \`\`(frame, record)\`\` where each
    first appears."""
    voxel = np.asarray(voxel)
    reach = np.ceil(within / voxel).astype(int)
    lo = np.maximum(np.round(mother["centroid"]).astype(int) - reach, 0).tolist()
    hi = np.minimum(np.round(mother["centroid"]).astype(int) + reach + 1, shape).tolist()
    found: list[tuple[int, dict]] = []
    for t in range(lost, min(lost + gap, frames)):
        for d in newborns(read(t - 1, lo, hi), read(t, lo, hi), lo, voxel, mother["centroid"], within, 0.1 * mother["volume"]):
            found.append((t, d))
            if len(found) == 2:
                return found
    return found


def lineage(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
            label: int, frames: int, margin, max_branches: int = 8) -> list[dict[int, dict]]:
    """The object labelled \`\`label\`\` at frame \`\`t0\`\` followed forward and back (\`\`follow\`\`),
    and forward through its divisions: the daughters of each nucleus that collapses and is
    lost are followed too. Returns branches (frame -> record), the first the object's own;
    each daughter's first record says \`\`mother\`\`, the branch it came from."""
    main = dict(follow(read, shape, voxel, t0, label, range(t0 + 1, frames), margin))
    main.update(follow(read, shape, voxel, t0, label, range(t0 - 1, -1, -1), margin))
    branches, todo = [main], [0]
    while todo and len(branches) < max_branches:
        k = todo.pop(0)
        last = max(branches[k])
        mother = mother_of(branches[k], last + 1)
        if mother is None or last + 1 >= frames:
            continue
        for t, d in daughters(read, shape, voxel, mother, last + 1, frames):
            if len(branches) >= max_branches:
                break
            branch = dict(follow(read, shape, voxel, t, d["label"], range(t + 1, frames), margin))
            branch[t]["mother"] = k
            branches.append(branch)
            todo.append(len(branches) - 1)
    return branches
`;const h=`https://cdn.jsdelivr.net/pyodide/v0.28.3/full/`,g={"chunkmirage/__init__.py":`"""chunkmirage's ops and fused stages, for the browser engine."""
`,"chunkmirage/core.py":n,"chunkmirage/fused.py":i,"chunkmirage/ops/__init__.py":a,"chunkmirage/ops/base.py":e,"chunkmirage/ops/pointwise.py":o,"chunkmirage/ops/filters.py":r,"chunkmirage/ops/segment.py":s,"chunkmirage/ops/combine.py":t,"chunkmirage/ops/terrain.py":c,"chunkmirage/cache.py":l,"chunkmirage/sources/__init__.py":`"""The computed sources, for the browser engine."""
`,"chunkmirage/sources/base.py":u,"chunkmirage/sources/synthetic.py":d,"chunkmirage/meshes.py":f,"chunkmirage/stitching.py":p,"chunkmirage/tracking.py":m},_=self;let v=null,y=null;async function b(e=[]){v=await(await import(
/* @vite-ignore */
`${h}pyodide.mjs`)).loadPyodide({indexURL:h}),await v.loadPackage([`numpy`,`pydantic`,...e]);for(let[e,t]of Object.entries(g))v.FS.mkdirTree(`/chunkmirage/${e.slice(0,e.lastIndexOf(`/`))}`),v.FS.writeFile(`/chunkmirage/${e}`,t);v.runPython(`import sys; sys.path.insert(0, "/chunkmirage")

import json
import numpy as np
from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec

VIEWS = {}
SOURCES = {}

def describe(url):
    from chunkmirage.sources.synthetic import open_synthetic
    ms = SOURCES.setdefault(url, open_synthetic(url))
    i = ms[0].info
    return json.dumps({
        "dtype": i.dtype.name, "channels": 1,
        "axes": [{"name": a, "unit": u} for a, u in zip(i.axes, i.units)],
        "levels": [{"shape": list(l.info.shape), "voxel": list(l.info.voxel_size), "origin": list(l.info.translation)} for l in ms],
    })

def _info(shape, dtype, chunk, voxel, origin=None, unit=""):
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")
    return ArrayInfo(shape=tuple(shape), dtype=np.dtype(dtype), chunk_shape=tuple(shape[: n - 3]) + tuple(chunk),
                     voxel_size=(1.0,) * (n - 3) + tuple(voxel), units=(unit,) * n, axes=axes,
                     translation=(0.0,) * (n - 3) + tuple(origin or (0.0,) * 3))

def plan(view, ops, shape, dtype, chunk, voxel, source=None):
    ops = [op_from_spec(s) for s in json.loads(ops)]
    info = _info(list(shape), dtype, list(chunk), list(voxel))
    if any(op.input_voxel_size() is not None for op in ops) or any(abs(r - 1) > 1e-9 for r in fused.scale(info, ops)):
        raise ValueError("this page runs ops on every level, each on its level's grid: one that reads at a "
                         "voxel size of its own, or changes the grid (downsample), runs from Python")
    out, lead, halo = fused.plan(info, ops)
    if source:
        describe(source)
    VIEWS[view] = (ops, dtype, list(chunk), source)
    added = list(out.shape[: out.ndim - len(halo)])  # leading axes the ops add (a model's channels)
    return json.dumps({"dtype": out.dtype.name, "lead": lead, "halo": list(halo), "ndim": out.ndim, "added": added})

def _mesh(m, result, box, shape, dtype, chunk, voxel, origin, unit):
    """legacy: a fragment (Neuroglancer's encoding). octree: from the whole coarsest level,
    the multi-resolution nodes (level, z, y, x each), header (count, fragment bytes,
    quantization bits) first, then the index. node: a node's mesh, header (vertices,
    triangles) then both as uint32, for the page to encode with Draco."""
    import struct
    from chunkmirage import meshes
    mode, levels = m.pop("mode", "legacy"), m.pop("levels", None)
    m.pop("raw", None)
    spec = meshes.MeshSpec(**{k: v for k, v in m.items() if k not in ("core_lo", "core_hi")})
    info = _info(list(shape), dtype, chunk, list(voxel), list(origin), unit)
    if mode == "mask":  # part of the coarsest level: inside or not, and the surface band, on
        # the part's core (the box the page asked for has a border of meshes.BAND around it)
        core = Box(tuple(m.pop("core_lo")), tuple(m.pop("core_hi")))
        grown = [min(a, meshes.BAND) for a in core.start]
        if any(g < meshes.BAND and b + g != a for g, a, b in zip(grown, core.start, box.start)):
            raise ValueError("a mask part needs a border of meshes.BAND voxels")
        inside = np.asarray(result) >= spec.threshold
        return np.ascontiguousarray(inside[core.slices()], np.uint8).tobytes() + meshes.surface_band(inside, core).astype(np.uint8).tobytes()
    if mode == "node":
        v, f = meshes.multires_fragment(spec, result, box, info, chunk)
        return struct.pack("<II", len(v), len(f)) + v.astype("<u4").tobytes() + f.astype("<u4").tobytes()
    return meshes.fragment(spec, result, box, info)

def octree(band, shape, mesh, chunk, unit):
    """A multi-resolution mesh's nodes (level, z, y, x each; header: their count, the
    fragments' size and the quantization bits) and index, from the whole coarsest level's
    surface band (uint8, 1 near the surface; meshes.surface_band)."""
    import struct
    from chunkmirage import meshes
    m = json.loads(mesh)
    m.pop("mode", None)
    levels = m.pop("levels")
    spec = meshes.MeshSpec(**{**m, "threshold": 1})  # the mask is inside or not
    infos = [_info(l["shape"], "uint8", list(chunk), l["voxel"], l["origin"], unit) for l in levels]
    block = np.frombuffer(band.to_py(), np.uint8).reshape(tuple(shape))
    lods = meshes.lod_levels(infos, spec)
    nodes = meshes.multires_nodes(spec, block, infos, list(chunk))
    flat = np.array([[lv, *n] for lv, ns in zip(lods, nodes) for n in ns], "<i4").reshape(-1, 4)
    head = struct.pack("<III", len(flat), meshes.FRAGMENT_BYTES, meshes.BITS)
    return head + flat.tobytes() + meshes.multires_index(nodes, infos, lods, list(chunk))

def compute(view, level, data, read_shape, in_lo, in_hi, out_lo, out_hi, full_shape, voxel, origin=None, mesh=None, unit=""):
    ops, dtype, chunk, source = VIEWS[view]
    full = tuple(full_shape)
    lead = len(full) - 3
    in_box = Box((0,) * lead + tuple(in_lo), full[:lead] + tuple(in_hi))
    if mesh is not None and json.loads(mesh).get("raw"):  # the page's own mask: no ops to run
        result = np.frombuffer(data.to_py(), np.uint8).reshape(tuple(read_shape))
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], "uint8", chunk, voxel, origin, unit)
    if data is None:  # a source computed here: the padded block, as a server stage reads it
        block = SOURCES[source][level].read_padded(in_box, edge=True)
    else:
        block = np.frombuffer(data.to_py(), dtype=np.dtype(dtype)).reshape(tuple(read_shape))
        block = fused.pad_edge(block, in_box, full)
    out, _, halo = fused.plan(_info(full, dtype, chunk, list(voxel)), ops)  # the level's voxels: for_level
    added = out.shape[: out.ndim - len(halo)]  # leading axes the ops add, whole in every chunk
    out_box = Box((0,) * len(added) + tuple(out_lo), tuple(added) + tuple(out_hi))
    result = fused.run(ops, block, in_box, out_box, out, halo)
    if mesh is not None:  # a mesh of these voxels, not a chunk
        return _mesh(json.loads(mesh), result, Box(tuple(out_lo), tuple(out_hi)), full[lead:], out.dtype.name, chunk, voxel, origin, unit)
    # a zarr chunk is always whole: one at the edge of the array is padded with the fill value
    result = np.pad(result, [(0, 0)] * len(added) + [(0, c - n) for c, n in zip(chunk, result.shape[len(added):])])
    return np.ascontiguousarray(result).astype(result.dtype.newbyteorder("<"), copy=False).tobytes()


def call(fn, args, arrays):
    """One step of a page's work in chunkmirage.stitching (the stitch page) or
    chunkmirage.tracking (track_*: the track page): JSON in, JSON (or a chunk's bytes) out,
    arrays as raw bytes."""
    if fn.startswith("track_"):
        return _track(fn, json.loads(args), arrays)
    from chunkmirage import stitching as S
    a = json.loads(args)
    p = S.StitchParams(**a.get("params", {}))
    if fn == "tiles":
        return json.dumps(S.tiles_from_bdv(a["xml"], a["base"], a["channel"]))
    if fn == "overlaps":  # each overlap, and the region of each of its two tiles that holds it
        out = []
        for i, j, lo, hi in S.overlaps(a["tiles"], p.margin):
            out.append({"tiles": [i, j], "lo": lo.tolist(), "hi": hi.tolist(),
                        "regions": [S.region(a["tiles"][t], p.level, lo, hi) for t in (i, j)]})
        return json.dumps(out)
    if fn == "points":
        block = np.frombuffer(arrays[0].to_py(), np.dtype(a["dtype"])).reshape(a["shape"])
        pts = S.points_in(a["tile"], p.level, block, a["start"], a["voxel"], a["lo"], a["hi"], p)
        return json.dumps(pts.round(3).tolist())
    if fn == "register":
        found = S.register(a["tiles"], a["points"], p, a.get("fixed", 0))
        found["grids"] = S.grids(a["tiles"], found["placements"])
        found["reference"] = S.compare(a["tiles"], found["placements"])
        return json.dumps(found)
    if fn == "regions":  # where each tile holds voxels [lo, hi) of a fused level
        lo, hi = S.scene_box(a["grid"], a["lo"], a["hi"])
        return json.dumps([S.region(t, a["level"], lo, hi, pl) for t, pl in zip(a["tiles"], a["placements"])])
    if fn == "fuse":
        blocks, k = [], 0
        for r in a["regions"]:
            if r is None:
                blocks.append(None)
                continue
            shape = [b - s for s, b in zip(*r)]
            blocks.append((np.frombuffer(arrays[k].to_py(), np.dtype(a["dtype"])).reshape(shape), r[0]))
            k += 1
        out = S.fuse(a["tiles"], a["placements"], a["level"], a["grid"], a["lo"], a["hi"], blocks, p.blend, a["dtype"])
        out = np.pad(out, [(0, c - n) for c, n in zip(a["chunk"], out.shape)])  # zarr chunks are whole
        return np.ascontiguousarray(out).astype(out.dtype.newbyteorder("<"), copy=False).tobytes()
    raise ValueError(f"no stitching step {fn}")

def _track(fn, a, arrays):
    from chunkmirage import tracking as T
    block = lambda k: np.frombuffer(arrays[k].to_py(), np.dtype(a["dtype"])).reshape(a["shape"])
    if fn == "track_measure":  # the object at its first frame, in a block from a.start
        return json.dumps(T.measure(block(0), a["label"], a["start"], a["voxel"]))
    if fn == "track_step":  # the object from the frame before (block 0) in the next (block 1)
        return json.dumps(T.step(block(0), a["label"], block(1), a["start"], a["voxel"]))
    if fn == "track_newborns":  # new labels near a mother lost to mitosis: blocks of the frame before and this
        return json.dumps(T.newborns(block(0), block(1), a["start"], a["voxel"], a["centre"], a["within"], a["min_volume"]))
    raise ValueError(f"no tracking step {fn}")
`)}async function x(e,t){v||await(y??=b(t));let n={};for(let[t,r]of Object.entries(e)){let e=JSON.parse(v.globals.get(`plan`)(t,JSON.stringify(r.ops),r.shape,r.dtype,r.chunk,r.voxel,r.source));if(e.halo.length!==3||e.added.length>1)throw Error(`view ${t}: a viewer shows three-axis volumes, with one channel axis at most; its ops leave ${e.ndim} axes`);n[t]=e}return n}async function S(e){v||await(y??=b());let t=(e.arrays??[]).map(e=>new Uint8Array(e)),n=await T(()=>v.globals.get(`call`)(e.fn,e.args,t));if(typeof n==`string`)return JSON.parse(n);let r=n.toJs();return n.destroy(),r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}const C={scipy:`scipy`,skimage:`scikit-image`},w=/* @__PURE__ */ new Map;async function T(e){for(let t=0;;t++)try{return e()}catch(e){let n=String(e?.message??``),r=(/No module named '(\w+)/.exec(n)??/`(\w+)` install you are using seems to be broken/.exec(n))?.[1];if(!r||!C[r]||t>1)throw e;w.has(r)||w.set(r,v.loadPackage([C[r]])),await w.get(r),v.runPython(`import sys\nfor m in [m for m in sys.modules if m == "${r}" or m.startswith("${r}.")]: del sys.modules[m]`)}}const E=`https://cdn.jsdelivr.net/gh/google/draco@1.5.7/javascript/`;let D=null;function O(){return D??=(async()=>{let e=await(await fetch(`${E}draco_encoder.js`)).text();return Function(`${e}\nreturn DracoEncoderModule;`)()({locateFile:e=>E+e})})()}async function k(e,t,n){let r=new DataView(e.buffer,e.byteOffset,8),i=r.getUint32(0,!0),a=r.getUint32(4,!0),o=new Uint8Array(t);if(!a)return o.buffer;let s=await O(),c=new Uint32Array(e.slice(8,8+i*12).buffer),l=new Uint32Array(e.slice(8+i*12).buffer),u=new s.Encoder,d=new s.MeshBuilder,f=new s.Mesh,p=new s.DracoInt8Array;try{d.AddFacesToMesh(f,a,l),d.AddFloatAttributeToMesh(f,s.POSITION,i,3,new Float32Array(c)),u.SetAttributeExplicitQuantization(s.POSITION,n,3,[0,0,0],2**n-1),u.SetSpeedOptions(3,3);let e=u.EncodeMeshToDracoBuffer(f,p);if(e>t)throw Error(`a mesh fragment took ${e} bytes, over ${t}`);for(let t=0;t<e;t++)o[t]=p.GetValue(t)}finally{s.destroy(p),s.destroy(f),s.destroy(d),s.destroy(u)}return o.buffer}async function A(e){let t=e.mesh?JSON.stringify({...e.mesh,size:void 0,bits:void 0}):void 0,n=await T(()=>v.globals.get(`compute`)(e.view,e.level,e.data?new Uint8Array(e.data):void 0,e.readShape,e.inLo,e.inHi,e.outLo,e.outHi,e.full,e.voxel,e.origin,t,e.unit??``)),r=n.toJs();return n.destroy(),e.mesh?.mode===`node`?k(r,e.mesh.size,e.mesh.bits):r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}_.onmessage=async({data:e})=>{try{if(e.type===`describe`)v||await(y??=b()),_.postMessage({reqId:e.reqId,value:JSON.parse(v.globals.get(`describe`)(e.source))});else if(e.type===`plan`)_.postMessage({reqId:e.reqId,value:await x(e.views,e.packages)});else if(e.type===`octree`){let t=await T(()=>v.globals.get(`octree`)(new Uint8Array(e.band),e.shape,JSON.stringify(e.mesh),e.chunk,e.unit)),n=t.toJs();t.destroy();let r=n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength);_.postMessage({reqId:e.reqId,value:r},[r])}else if(e.type===`call`){let t=await S(e);_.postMessage({reqId:e.reqId,value:t},t instanceof ArrayBuffer?[t]:[])}else if(e.type===`compute`){let t=await A(e);_.postMessage({reqId:e.reqId,value:t},[t])}}catch(t){let n=t?.message??t?.name??JSON.stringify(t);_.postMessage({reqId:e.reqId,error:String(n)})}};