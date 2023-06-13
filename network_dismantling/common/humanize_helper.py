# EDITED FROM THE HUMANIZE PACKAGE!
# from humanize.i18n import gettext as _
# from humanize.i18n import gettext_noop as N_
# from humanize.i18n import pgettext as P_

from fastnumbers import fast_real
from humanize import i18n

_ = i18n.gettext_module.gettext
N_ = lambda x: x

powers = [10 ** x for x in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]
human_powers = (
    N_("K"),
    N_("M"),
    N_("B"),
    N_("trillion"),
    N_("quadrillion"),
    N_("quintillion"),
    N_("sextillion"),
    N_("septillion"),
    N_("octillion"),
    N_("nonillion"),
    N_("decillion"),
    N_("googol"),
)

power_human_mapping = {p: h for p, h in zip(powers, human_powers)}
human_power_mapping = {h: p for p, h in zip(powers, human_powers)}


def intword(value, format="%.1f"):
    """Converts a large integer to a friendly text representation.
    Works best for numbers over 1 million. For example, 1000000 becomes "1.0 million",
    1200000 becomes "1.2 million" and "1200000000" becomes "1.2 billion". Supports up to
    decillion (33 digits) and googol (100 digits).
    Args:
        value (int, float, string): Integer to convert.
        format (str): to change the number of decimal or general format of the number
            portion.
    Returns:
        str: friendly text representation as a string, unless the value passed could not
        be coaxed into an int.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value

    if value < powers[0]:
        return str(value)

    for ordinal, power in enumerate(powers[1:], 1):
        if value < power:
            chopped = value / float(powers[ordinal - 1])
            if float(format % chopped) == float(10 ** 3):
                chopped = value / float(powers[ordinal])
                human_power = _(human_powers[ordinal])
            else:
                human_power = _(human_powers[ordinal - 1])

            human_value = format % chopped

            return "{}{}".format(human_value.rstrip('0').rstrip('.'), human_power)

    return str(value)


def from_human(value):
    """Converts a humanized number back to the correct number type"""

    if isinstance(value, str):
        value = value.strip()
        for human_power, real_power in human_power_mapping.items():
            if human_power.lower() in value[-len(human_power):].lower():
                value = fast_real(value.replace(human_power, '')) * real_power
                break
    else:
        value = fast_real(value, raise_on_invalid=True)

    return value
