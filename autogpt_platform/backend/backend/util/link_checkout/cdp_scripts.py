"""Functions the checkout evaluates in a page's isolated world, never in the
page's own: the page can't override what they call or see what they return."""

# The autofill field names (the last token of ``autocomplete``) each card role
# must declare. Payment processors' card frames and merchants' own card forms
# set them so browsers can autofill; nothing else on a page uses them.
CARD_AUTOCOMPLETE = {
    "number": ["cc-number"],
    "cvc": ["cc-csc"],
    "expiry": ["cc-exp"],
    "exp_month": ["cc-exp-month"],
    "exp_year": ["cc-exp-year"],
}

# The one element a selector matches. A selector the page cannot parse (an
# agent's Playwright-only :visible, say) is reported rather than thrown, so the
# agent is told what to fix instead of getting a generic failure.
FIND_ONE = (
    "function(selector) { let nodes; try { nodes = document.querySelectorAll(selector); }"
    " catch (error) { return 'invalid_selector'; }"
    " return nodes.length === 1 ? nodes[0] : null; }"
)
CHECK_CONTROL = r"""function(names, allowDisabled) {
    const box = this.getBoundingClientRect();
    const style = getComputedStyle(this);
    if (!this.isConnected || box.width <= 0 || box.height <= 0 ||
        style.visibility !== 'visible' || style.display === 'none' ||
        (this.disabled && !allowDisabled)) return 'not_ready';
    if (names === null) {
        const button = this instanceof HTMLButtonElement ||
            (this instanceof HTMLInputElement &&
             ['submit', 'button', 'image'].includes(this.type));
        return button ? 'ok' : 'not_card_field';
    }
    if (!(this instanceof HTMLInputElement) || this.type === 'hidden' ||
        this.readOnly) return 'not_card_field';
    const tokens = (this.getAttribute('autocomplete') || '').trim().toLowerCase()
        .split(/\s+/);
    if (!names.includes(tokens[tokens.length - 1])) return 'not_card_field';
    return this.value === '' ? 'ok' : 'not_ready';
}"""

# The Link Pay Token field (``pay_token``): an input named for it, still empty.
# It is often visually hidden, so unlike a card field it needn't be visible.
CHECK_TOKEN_FIELD = r"""function() {
    if (!this.isConnected || !(this instanceof HTMLInputElement) ||
        this.name !== 'link_pay_token') return 'not_card_field';
    return this.value === '' ? 'ok' : 'not_ready';
}"""
INJECT_PAY_TOKEN = r"""function(token) {
    const el = this;
    if (!el.isConnected || !(el instanceof HTMLInputElement) ||
        el.name !== 'link_pay_token' || el.value !== '') return false;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')
        .set.call(el, token);
    el.dispatchEvent(new Event('input', {bubbles: true}));
    return true;
}"""
