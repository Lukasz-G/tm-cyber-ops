"""Readable names for Drebin features (APIGraph's extraction).

A raw feature name is `<set>_<value>`, all lower case, for example
`requestedpermissionlist_android.permission.read_phone_state`. `label()` returns the set as a short
category and the value in words: ("Asks for permission", "phone identity and call state").

The texts say what a feature is (a permission, an API call, a component of a known SDK). They do not
say whether it is malicious.
"""

from __future__ import annotations

CATEGORIES = {
    "requestedpermissionlist": "Asks for permission",
    "usedpermissionslist": "Uses permission",
    "suspiciousapilist": "Calls sensitive API",
    "restrictedapilist": "Calls protected API",
    "intentfilterlist": "Reacts to event",
    "broadcastreceiverlist": "Background receiver",
    "servicelist": "Background service",
    "activitylist": "Screen",
    "contentproviderlist": "Data provider",
    "hardwarecomponentslist": "Needs device feature",
    "urldomainlist": "Contacts address",
}

PERMISSIONS = {
    "read_phone_state": "phone identity and call state",
    "send_sms": "send SMS",
    "receive_sms": "receive SMS",
    "read_sms": "read SMS",
    "write_sms": "write SMS",
    "receive_mms": "receive MMS",
    "internet": "internet access",
    "access_fine_location": "precise location",
    "access_coarse_location": "approximate location",
    "access_mock_location": "fake locations",
    "receive_boot_completed": "start when the phone boots",
    "read_contacts": "read contacts",
    "write_contacts": "write contacts",
    "call_phone": "make phone calls",
    "process_outgoing_calls": "see outgoing calls",
    "read_logs": "read system logs",
    "vibrate": "vibrate",
    "wake_lock": "keep the phone awake",
    "get_tasks": "see running apps",
    "access_wifi_state": "see Wi-Fi state",
    "change_wifi_state": "change Wi-Fi state",
    "access_network_state": "see network state",
    "change_network_state": "change network state",
    "write_external_storage": "write shared storage",
    "read_external_storage": "read shared storage",
    "install_shortcut": "add home-screen shortcuts",
    "read_history_bookmarks": "read browser bookmarks",
    "write_history_bookmarks": "write browser bookmarks",
    "set_wallpaper": "set the wallpaper",
    "bluetooth": "Bluetooth",
    "broadcast_sticky": "send sticky broadcasts",
    "camera": "camera",
    "record_audio": "microphone",
    "get_accounts": "list accounts",
    "install_packages": "install apps",
    "delete_packages": "remove apps",
    "system_alert_window": "draw over other apps",
    "disable_keyguard": "disable the lock screen",
    "mount_unmounted_filesystems": "mount storage",
    "write_settings": "change system settings",
    "restart_packages": "stop other apps",
    "kill_background_processes": "stop background apps",
    "receive": "push messages (Google C2DM)",
}

APIS = {
    "getdeviceid": "device ID (IMEI)",
    "getsubscriberid": "SIM subscriber ID (IMSI)",
    "getline1number": "phone number",
    "getsimserialnumber": "SIM serial number",
    "getsimoperator": "SIM operator",
    "getnetworkoperator": "network operator",
    "getcelllocation": "cell location",
    "sendtextmessage": "send SMS",
    "sendmultiparttextmessage": "send long SMS",
    "getwifistate": "Wi-Fi state",
    "getconnectioninfo": "Wi-Fi connection info",
    "getsystemservice": "system services",
    "httppost": "HTTP POST request",
    "httpget": "HTTP GET request",
    "exec": "run a shell command",
    "getaccounts": "list accounts",
    "getinstalledpackages": "list installed apps",
    "getrunningtasks": "list running apps",
    "requestlocationupdates": "location updates",
    "getlastknownlocation": "last known location",
    "getprovider": "location provider",
    "notify": "show a notification",
    "vibrate": "vibrate",
    "abortbroadcast": "block a broadcast (for example an incoming SMS)",
    "loadlibrary": "load native code",
    "getmacaddress": "MAC address",
    "getactivenetworkinfo": "active network",
    "setwifienabled": "switch Wi-Fi on or off",
    "getpackageinfo": "app package details",
    "getmessagebody": "read SMS text",
    "getoriginatingaddress": "SMS sender",
    "getinstalledapplications": "list installed apps",
    "isactivenetworkmetered": "metered network check",
}

EVENTS = {
    "boot_completed": "phone finished booting",
    "sms_received": "SMS received",
    "data_sms_received": "data SMS received",
    "phone_state": "call state changed",
    "new_outgoing_call": "outgoing call",
    "package_added": "app installed",
    "package_removed": "app removed",
    "package_replaced": "app updated",
    "user_present": "phone unlocked",
    "screen_on": "screen switched on",
    "screen_off": "screen switched off",
    "connectivity_change": "network changed",
    "battery_changed_action": "battery level changed",
    "sig_str": "signal strength changed",
    "main": "app start (launcher entry)",
    "send": "share target",
    "create_shortcut": "shortcut created",
    "media_eject": "storage ejected",
    "media_removed": "storage removed",
    "wallpaperservice": "live wallpaper",
    "in_app_notify": "in-app billing notice",
    "purchase_state_changed": "in-app purchase changed",
    "response_code": "in-app billing response",
    "install_referrer": "install referral",
}

# component name prefixes of widely used SDKs
SDKS = {
    "com.airpush.": "Airpush ad SDK",
    "com.google.ads.": "Google Ads SDK",
    "com.google.android.gms.ads.": "Google Ads SDK",
    "com.millennialmedia.": "Millennial Media ad SDK",
    "com.tapjoy.": "Tapjoy ad SDK",
    "com.adwo.": "Adwo ad SDK",
    "net.youmi.": "Youmi ad SDK",
    "com.mobfox.": "MobFox ad SDK",
    "com.greystripe.": "Greystripe ad SDK",
    "com.pad.android.": "LeadBolt ad SDK",
    "com.waps.": "Waps ad SDK",
    "com.admob.": "AdMob SDK",
    "com.inmobi.": "InMobi ad SDK",
    "com.flurry.": "Flurry analytics SDK",
    "com.paypal.": "PayPal SDK",
    "com.scoreloop.": "Scoreloop games SDK",
    "com.papaya.": "Papaya social SDK",
    "com.google.android.c2dm.": "Google push messaging",
    "com.google.update.": "a component named 'Google update'",
    "ad.notify.": "ad notification component",
}


def _words(s: str) -> str:
    return s.replace("_", " ").strip()


def _component(value: str) -> str:
    for prefix, sdk in SDKS.items():
        if value.startswith(prefix):
            return f"{value.rsplit('.', 1)[-1]} ({sdk})"
    if value.startswith(".") or "." not in value:
        return f"{value.lstrip('.')} (the app's own)"
    return value


def label(feature: str) -> tuple[str, str]:
    """(category, readable value) of a raw Drebin feature name."""
    kind, _, value = feature.partition("_")
    category = CATEGORIES.get(kind, kind)
    if kind in ("requestedpermissionlist", "usedpermissionslist"):
        name = value.rsplit(".", 1)[-1]
        return category, PERMISSIONS.get(name, _words(name))
    if kind in ("suspiciousapilist", "restrictedapilist"):
        tail = value.replace(";->", ".").rsplit("/", 1)[-1]
        method = tail.rsplit(".", 1)[-1]
        cls = tail.rsplit(".", 2)[-2] if "." in tail else ""
        if method == "<init>":
            return category, f"creates {cls}"
        if method in APIS:
            return category, APIS[method]
        if cls and len(cls) > 24:
            cls = cls[:22] + "…"
        return category, f"{method} ({cls})" if cls else method
    if kind == "intentfilterlist":
        name = value.rsplit(".", 1)[-1]
        return category, EVENTS.get(name, _words(name))
    if kind in ("broadcastreceiverlist", "servicelist", "activitylist", "contentproviderlist"):
        return category, _component(value)
    if kind == "hardwarecomponentslist":
        return category, _words(value.split(".", 2)[-1].replace(".", " "))
    return category, value
