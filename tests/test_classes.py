from parsel.utils.classes import Taxonomy


def test_main_class_ids_are_contiguous():
    tax = Taxonomy()
    assert [c.id for c in tax.classes] == list(range(9))


def test_every_subclass_has_valid_parent():
    tax = Taxonomy()
    parents = {c.id for c in tax.classes}
    assert all(s.parent in parents for s in tax.subclasses)


def test_every_main_class_has_a_subclass():
    tax = Taxonomy()
    assert {s.parent for s in tax.subclasses} == {c.id for c in tax.classes}


def test_subclass_ids_unique_and_lut_shape():
    tax = Taxonomy()
    ids = [s.id for s in tax.subclasses]
    assert len(ids) == len(set(ids))
    assert tax.subclass_to_class().shape[0] == max(ids) + 1


def test_palette_shape():
    tax = Taxonomy()
    assert tax.palette().shape == (9, 3)
    assert tax.palette("subclasses").shape == (19, 3)
