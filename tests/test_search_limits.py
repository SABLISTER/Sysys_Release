from gui.search_runner import (
    ProviderChoice,
    _per_query_limit,
    normalize_provider_choices,
    provider_result_limits,
    selected_search_provider_count,
)


def test_selected_provider_count_expands_all():
    assert selected_search_provider_count(ProviderChoice.ALL) == 6
    assert selected_search_provider_count([ProviderChoice.ALL]) == 6


def test_provider_result_limits_split_total_across_selected_engines():
    providers = [
        ProviderChoice.PUBMED,
        ProviderChoice.S2,
        ProviderChoice.OPENALEX,
    ]

    limits = provider_result_limits(300, providers)

    assert limits == {
        ProviderChoice.PUBMED: 100,
        ProviderChoice.S2: 100,
        ProviderChoice.OPENALEX: 100,
    }
    assert sum(limits.values()) == 300


def test_provider_result_limits_keep_remainder_in_ui_order():
    providers = [
        ProviderChoice.PUBMED,
        ProviderChoice.S2,
        ProviderChoice.OPENALEX,
    ]

    limits = provider_result_limits(302, providers)

    assert limits[ProviderChoice.PUBMED] == 101
    assert limits[ProviderChoice.S2] == 101
    assert limits[ProviderChoice.OPENALEX] == 100
    assert sum(limits.values()) == 302


def test_provider_result_limits_can_assign_zero_when_cap_is_tiny():
    providers = [
        ProviderChoice.PUBMED,
        ProviderChoice.S2,
        ProviderChoice.OPENALEX,
    ]

    limits = provider_result_limits(2, providers)

    assert limits[ProviderChoice.PUBMED] == 1
    assert limits[ProviderChoice.S2] == 1
    assert limits[ProviderChoice.OPENALEX] == 0
    assert sum(limits.values()) == 2


def test_all_provider_limits_sum_to_total():
    limits = provider_result_limits(500, ProviderChoice.ALL)

    assert set(limits) == normalize_provider_choices(ProviderChoice.ALL)
    assert sum(limits.values()) == 500


def test_provider_budget_then_query_budget_divides_in_two_stages():
    providers = [ProviderChoice.S2, ProviderChoice.OPENALEX]
    provider_budget = provider_result_limits(100, providers)[ProviderChoice.S2]

    assert provider_budget == 50
    assert _per_query_limit(provider_budget, 4) == 13
