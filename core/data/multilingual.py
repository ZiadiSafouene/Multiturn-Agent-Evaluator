"""
core.data.multilingual
~~~~~~~~~~~~~~~~~~~~~~

Lexical resources, standard intent markers, and phrase templates for the
three supported languages: English, Arabic, and French.
"""

from __future__ import annotations

from typing import Dict, List

from core.utils.models import Language

SUPPORTED_LANGUAGES: List[Language] = [
    Language.ENGLISH,
    Language.ARABIC,
    Language.FRENCH,
]

LANGUAGE_NAMES: Dict[Language, str] = {
    Language.ENGLISH: "English",
    Language.ARABIC: "العربية (Modern Standard Arabic)",
    Language.FRENCH: "Français",
}

AFFIRMATIVES: Dict[Language, List[str]] = {
    Language.ENGLISH: ["yes", "yeah", "correct", "sure", "confirm", "proceed", "okay"],
    Language.ARABIC: ["نعم", "صحيح", "أكيد", "بالتأكيد", "موافق", "تمام"],
    Language.FRENCH: ["oui", "ouais", "d'accord", "exactement", "tout à fait", "confirmer"],
}

NEGATIVES: Dict[Language, List[str]] = {
    Language.ENGLISH: ["no", "nope", "incorrect", "cancel", "abort", "stop"],
    Language.ARABIC: ["لا", "كلا", "غير صحيح", "إلغاء", "توقف"],
    Language.FRENCH: ["non", "pas du tout", "incorrect", "annuler", "arrêter"],
}

GREETINGS: Dict[Language, List[str]] = {
    Language.ENGLISH: ["hello", "hi", "good morning", "good afternoon"],
    Language.ARABIC: ["مرحبا", "أهلا وسهلا", "السلام عليكم", "صباح الخير"],
    Language.FRENCH: ["bonjour", "salut", "bonsoir"],
}

COMMON_BANKING_ALIASES: Dict[Language, Dict[str, List[str]]] = {
    Language.ENGLISH: {
        "account_number": ["account id", "account", "account no", "acc num"],
        "balance": ["funds", "money", "current balance"],
        "transfer_amount": ["amount", "sum", "quantity"],
    },
    Language.ARABIC: {
        "account_number": ["رقم الحساب", "حسابي", "الحساب"],
        "balance": ["الرصيد", "كم معي", "رصيد الحساب"],
        "transfer_amount": ["المبلغ", "القيمة", "مبلغ التحويل"],
    },
    Language.FRENCH: {
        "account_number": ["numéro de compte", "num de compte", "compte"],
        "balance": ["solde", "mon solde", "fonds"],
        "transfer_amount": ["montant", "somme", "valeur du virement"],
    },
}
