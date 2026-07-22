from dataclasses import replace

from PIL import Image

from app.config import Settings
from app.restoration.codeformer_service import FaceCandidate, FaceCandidateResult
from app.restoration.face_compositor import FaceSoftMaskCompositor


def _candidate(index: int = 0, *, accepted: bool = True, affine_matrix=None) -> FaceCandidate:
    return FaceCandidate(
        face_index=index,
        bbox=(0.0, 0.0, 64.0, 64.0),
        detection_score=0.9,
        original_face=Image.new("RGB", (64, 64), "black"),
        restored_face=Image.new("RGB", (64, 64), "white"),
        affine_matrix=affine_matrix or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        landmarks=(),
        identity_score=0.9,
        identity_accepted=accepted,
        landmark_accepted=accepted,
        selected=accepted,
    )


def test_compositor_does_not_paste_unselected_candidate():
    source = Image.new("RGB", (64, 64), "black")
    candidate = replace(_candidate(), selected=False)

    composed = FaceSoftMaskCompositor(Settings(_env_file=None)).composite(
        FaceCandidateResult(source, (candidate,), 1)
    )

    assert composed.pasted_face_count == 0
    assert composed.image.tobytes() == source.tobytes()


def test_compositor_leaves_image_unchanged_when_candidate_is_rejected():
    source = Image.new("RGB", (64, 64), "black")
    result = FaceCandidateResult(source, (_candidate(accepted=False),), 1)

    composed = FaceSoftMaskCompositor(Settings(_env_file=None)).composite(result)

    assert composed.image.tobytes() == source.tobytes()
    assert composed.pasted_face_count == 0


def test_compositor_uses_soft_mask_with_opaque_center_and_preserved_corners():
    source = Image.new("RGB", (64, 64), "black")
    settings = Settings(
        _env_file=None,
        face_mask_inset_ratio=0.12,
        face_mask_blur_ratio=0.08,
        face_mask_opacity=1.0,
        face_color_match_enabled=False,
        face_texture_blend=0.0,
    )

    composed = FaceSoftMaskCompositor(settings).composite(
        FaceCandidateResult(source, (_candidate(),), 1)
    )

    assert composed.pasted_face_count == 1
    assert composed.image.getpixel((32, 32))[0] > 245
    assert composed.image.getpixel((0, 0)) == (0, 0, 0)
    edge_value = composed.image.getpixel((8, 32))[0]
    assert 0 < edge_value < 255


def test_compositor_respects_source_to_aligned_affine_matrix():
    source = Image.new("RGB", (128, 64), "black")
    # source x=64..127 maps to aligned crop x=0..63.
    candidate = _candidate(affine_matrix=((1.0, 0.0, -64.0), (0.0, 1.0, 0.0)))

    composed = FaceSoftMaskCompositor(
        Settings(
            _env_file=None,
            face_mask_opacity=1.0,
            face_color_match_enabled=False,
            face_texture_blend=0.0,
        )
    ).composite(FaceCandidateResult(source, (candidate,), 1))

    assert composed.image.getpixel((96, 32))[0] > 245
    assert composed.image.getpixel((32, 32)) == (0, 0, 0)


def test_compositor_skips_invalid_affine_but_pastes_other_faces():
    source = Image.new("RGB", (64, 64), "black")
    invalid = replace(_candidate(0), affine_matrix=((1.0, 0.0),))
    valid = _candidate(1)

    composed = FaceSoftMaskCompositor(Settings(_env_file=None)).composite(
        FaceCandidateResult(source, (invalid, valid), 2)
    )

    assert composed.pasted_face_count == 1


def test_compositor_matches_face_tone_and_retains_source_texture():
    original = Image.new("RGB", (64, 64), (165, 112, 88))
    for x in range(8, 56, 4):
        original.putpixel((x, 32), (115, 75, 60))
    candidate = replace(
        _candidate(),
        original_face=original,
        restored_face=Image.new("RGB", (64, 64), (235, 205, 190)),
    )
    settings = Settings(
        _env_file=None,
        face_color_match_enabled=True,
        face_color_match_strength=1.0,
        face_texture_blend=0.35,
    )

    prepared = FaceSoftMaskCompositor(settings)._prepare_restored_face(candidate)

    assert prepared.getpixel((32, 32))[0] < 220
    assert prepared.getpixel((8, 32)) != prepared.getpixel((9, 32))