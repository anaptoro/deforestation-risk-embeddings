from src.modeling.export_weights import weights_csv, gee_fragment


def test_weights_csv_basic():
    w = [1.0, 2.5, -0.125]
    s = weights_csv(w)
    assert s == "1,2.5,-0.125"


def test_gee_fragment_contains_required_fields():
    w = [0.1] * 66
    frag = gee_fragment(
        w=w,
        b=-0.42,
        title="Risk 66 v5",
        tag="logit66",
        year=2022,
        lat=-9.5,
        lon=-62.5,
        zoom=9,
    )

    # must contain these keys
    assert "title=" in frag
    assert "year=2022" in frag
    assert "lat=-9.5" in frag
    assert "lon=-62.5" in frag
    assert "zoom=9" in frag

    # weights always present
    assert "w=" in frag
    assert "b=-0.42" in frag

    # your fragments end with a trailing ;
    assert frag.endswith(";")
