import pandas as pd


FILE_PATH = "data/formal_evaluation_set.csv"

df = pd.read_csv(FILE_PATH)

df["issue_category"] = df["issue_category"].astype("object")
df["specific_issue"] = df["specific_issue"].astype("object")
df["sentiment"] = df["sentiment"].astype("object")
df["severity"] = df["severity"].astype("object")


annotations = {
    "4c9a177c-9528-4586-873a-e6df77509f50": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "bcf33e01-1d52-4c70-88ca-c743148854d6": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Lack of privacy protection",
        "sentiment": "Negative",
        "severity": "High"
    },

    "f385579a-1f55-4822-b29c-c846b486e216": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "a680c918-8c99-47ac-8fcd-124f070894e5": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "3a3ab103-1e89-4f5b-9244-837fb5814010": {
        "issue_category": "Other / General",
        "specific_issue": "Unclear messaging dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "649b8c78-aa3b-4de4-88d2-27311c726658": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to sign up",
        "sentiment": "Negative",
        "severity": "High"
    },

    "7befd1a1-4af0-442a-a95b-c0946f99d1ae": {
        "issue_category": "Account & Login",
        "specific_issue": "Verification code not received promptly",
        "sentiment": "Negative",
        "severity": "High"
    },

    "06b163bc-ab79-44ef-8a23-a73de5553cd5": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction and app abandonment",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "3e5a4639-e087-476c-bf5f-e3206b41018e": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Suspected unauthorized account access",
        "sentiment": "Negative",
        "severity": "High"
    },

    "08dd3c9c-1666-4d23-bfe5-083e62737c47": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

     "a9345085-07c7-4361-8c23-6c46427ab035": {
        "issue_category": "Account & Login",
        "specific_issue": "Verification process and geolocation issue",
        "sentiment": "Negative",
        "severity": "High"
    },

    "43453c65-f502-4129-ab8a-aa2c38ecdc2d": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "6453fbd7-c320-4fe2-9c38-7a4f78f45961": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Privacy concerns and poor app performance",
        "sentiment": "Negative",
        "severity": "High"
    },

    "86e68aa6-2acf-4446-8b60-fb5bf40b3a0a": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Unable to install app",
        "sentiment": "Negative",
        "severity": "High"
    },

    "30ff2783-a564-4686-af2b-f243dd174cfd": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to log in despite repeated attempts",
        "sentiment": "Negative",
        "severity": "High"
    },

    "48717881-514c-464f-b1dc-b1412126124e": {
        "issue_category": "Feature Problem",
        "specific_issue": "App shortcut not visible",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "1026e598-3f9d-423d-a527-9ab9b7ad500d": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to unsend all messages",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "837023d2-9d56-4588-87c7-06a6433f751d": {
        "issue_category": "Account & Login",
        "specific_issue": "Difficulty logging in",
        "sentiment": "Negative",
        "severity": "High"
    },

     "ebb08e43-f6b6-4ae0-87ca-6ed57a6a152a": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "94ec71aa-043a-486a-bec0-f7bfcd5f0c34": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "24fe1ac8-a36b-462d-b615-ba57e5702361": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "b984c22e-fe53-4c74-87f3-adc260d44fd7": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to restore old Bitmoji appearance",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "e127082b-68a3-4ad5-85c9-121bfce119ce": {
        "issue_category": "Account & Login",
        "specific_issue": "Lost account access after changing phone",
        "sentiment": "Negative",
        "severity": "High"
    },

    "142c01dd-ea22-4615-8686-afbf38006bf0": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Spyware and privacy concern",
        "sentiment": "Negative",
        "severity": "High"
    },

    "bf37909a-3f46-4fb4-aa1a-601bf3aa75bf": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "da520ed2-fe10-4ff2-9177-31060b4e23f3": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },
    "a56bcf01-bdd1-4b8c-9ceb-2c7d575315ec": {
        "issue_category": "Feature Problem",
        "specific_issue": "Hidden status no longer visible after update",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "9e8bd8f5-ab30-43a4-8e10-a132d548385b": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "d81bf2fe-7d25-4ebb-ac58-fb5ad5cc0e61": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Multiple problems with desktop version",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "f344d84b-0e34-42e6-b2d7-61268466bf06": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "d6f6bb79-87ea-49e0-8c05-b7ac3461abec": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "2fb821a5-48b0-4d3b-b59e-62d330edb4bd": {
        "issue_category": "Feature Problem",
        "specific_issue": "Notifications not working when app is closed",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "2df1d5ca-0fd0-4381-b616-0e5d85c59e99": {
        "issue_category": "Account & Login",
        "specific_issue": "Account banned without clear reason",
        "sentiment": "Negative",
        "severity": "High"
    },

    "9dac3d93-b3b7-4acc-962f-25cea0047b77": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction with app updates",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "721016a5-929b-4e63-89b4-8f1e01d57945": {
        "issue_category": "Account & Login",
        "specific_issue": "Login code not received by email",
        "sentiment": "Negative",
        "severity": "High"
    },

    "efac743d-b89f-44bc-9d22-d9b83facd6de": {
        "issue_category": "Feature Request",
        "specific_issue": "Request for voice and video calling",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "bfc08ee1-2e3e-48e5-abdd-d7f470c12edb": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "20e2d056-3d3c-4c3c-8a0e-00c7a52a548d": {
        "issue_category": "Other / General",
        "specific_issue": "General unrelated comment",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "d2bb94b3-6b4f-42a2-8129-4f8de2b972de": {
        "issue_category": "Feature Problem",
        "specific_issue": "Error playing voicemail",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "72fb9f92-f18b-45b7-8daa-e9fa2d128613": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to access existing account",
        "sentiment": "Negative",
        "severity": "High"
    },

    "d58c342e-1ff6-4210-8286-b1182e01dd0e": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to log in for several days",
        "sentiment": "Negative",
        "severity": "High"
    },

    "c9197c24-f4f8-489c-b30e-2e652a53af54": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to remove block feature",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "b7d854f9-2466-4787-a204-de6e6bb3b507": {
        "issue_category": "Feature Request",
        "specific_issue": "Request for latest Messenger update",
        "sentiment": "Neutral",
        "severity": "Low"
    },

     "721016a5-929b-4e63-89b4-8f1e01d57945": {
        "issue_category": "Account & Login",
        "specific_issue": "Login code not received by email",
        "sentiment": "Negative",
        "severity": "High"
    },

    "efac743d-b89f-44bc-9d22-d9b83facd6de": {
        "issue_category": "Feature Request",
        "specific_issue": "Request for voice and video calling",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "bfc08ee1-2e3e-48e5-abdd-d7f470c12edb": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "20e2d056-3d3c-4c3c-8a0e-00c7a52a548d": {
        "issue_category": "Other / General",
        "specific_issue": "General unrelated comment",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "d2bb94b3-6b4f-42a2-8129-4f8de2b972de": {
        "issue_category": "Feature Problem",
        "specific_issue": "Error playing voicemail",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "72fb9f92-f18b-45b7-8daa-e9fa2d128613": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to access existing account",
        "sentiment": "Negative",
        "severity": "High"
    },

    "d58c342e-1ff6-4210-8286-b1182e01dd0e": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to log in for several days",
        "sentiment": "Negative",
        "severity": "High"
    },

    "c9197c24-f4f8-489c-b30e-2e652a53af54": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to remove block feature",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "b7d854f9-2466-4787-a204-de6e6bb3b507": {
        "issue_category": "Feature Request",
        "specific_issue": "Request for latest Messenger update",
        "sentiment": "Neutral",
        "severity": "Low"
    },

        "c6110b5d-fea2-40cd-9cfa-21ce085a5cd3": {
        "issue_category": "Feature Problem",
        "specific_issue": "Unable to access older messages",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "e9c10f24-f709-4234-aeed-b6e8470b7991": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "593a28d5-cbb8-418c-9fce-dd8357b1d572": {
        "issue_category": "Privacy / Security",
        "specific_issue": "General privacy and confidentiality praise",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "e324e9cd-1f70-4489-8b44-1426402f4c48": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to sign in successfully",
        "sentiment": "Negative",
        "severity": "High"
    },

    "6f5e1d41-8bae-47b1-8df3-19ed82581950": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "789ff14d-59be-46d5-a2d8-464729303b43": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "61a1041c-b423-4e6f-bde4-1c24aa1c9d32": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to access Telegram account",
        "sentiment": "Negative",
        "severity": "High"
    },

    "aa31114c-c717-4df2-8739-d30ec606ed4a": {
        "issue_category": "Other / General",
        "specific_issue": "General mixed feedback",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "2c6683ac-89e6-4e5b-9757-613958b6139d": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on filters",
        "sentiment": "Positive",
        "severity": "Low"
    },
     "7844ea7f-c245-4ee9-bc3a-e04e9f554241": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to remove or hide new updates feature",
        "sentiment": "Negative",
        "severity": "Low"
    },
        "9a2d809c-776d-4539-bbe1-607ca63465fb": {
        "issue_category": "Feature Problem",
        "specific_issue": "New features not visible after update",
        "sentiment": "Negative",
        "severity": "Medium"
    },
        "7c55539b-e590-453d-b4e5-c61f136e20fe": {
        "issue_category": "Feature Problem",
        "specific_issue": "Excessive notifications and advertising",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "4e3c361c-4d54-4e5f-b9f7-ad8026d31c6a": {
        "issue_category": "Performance",
        "specific_issue": "App is very slow",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "744ebf79-9b04-4bed-bcec-8df869e4cac3": {
        "issue_category": "Performance",
        "specific_issue": "App stuck on connecting for days",
        "sentiment": "Negative",
        "severity": "High"
    },

    "0dd44ca9-21bc-46cd-9fbd-2f62428d19f2": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "6d3fa7e0-ed33-4e55-a3cc-5ca77edb9708": {
        "issue_category": "Account & Login",
        "specific_issue": "Login blocked by unofficial app warning",
        "sentiment": "Negative",
        "severity": "High"
    },

    "f663ef5d-926a-45de-b444-81eb63853cee": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "d0414e48-1baa-4b0e-aa74-612ba1ed402b": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "b9899983-e3e1-4175-9c93-be1c9e6c0b28": {
        "issue_category": "Account & Login",
        "specific_issue": "SMS verification not working",
        "sentiment": "Negative",
        "severity": "High"
    },
    "95133686-8dfc-48b8-8cad-788e53c6ac19": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },
       "ef462d1c-bd7c-44c0-abe3-29b325731478": {
        "issue_category": "Feature Problem",
        "specific_issue": "Too many advertisements",
        "sentiment": "Negative",
        "severity": "Medium"
    },
     "498e0006-019c-435a-b65f-f8f850ce9366": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction without specific issue",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "72f7d601-1524-445c-81a9-8341146bd8dc": {
        "issue_category": "Performance",
        "specific_issue": "Slow update and excessive storage usage",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "6b6e3ec8-aa79-479c-8fc0-703dd8292961": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on filters",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "13a12324-63c4-4584-a6ea-ed1489a6a199": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "fea78dbc-4a0e-4d2a-a734-c40b66513bfd": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to remove or hide Channels section",
        "sentiment": "Negative",
        "severity": "Low"
    },

     "b31d9668-02ee-4b1a-a155-c946d15f9eed": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "App update not installing",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "7ebf2c4f-20aa-4723-a34d-f2a37850f660": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "App freezes while creating a poll",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "8587854c-ed70-420e-982e-85b918b8d498": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "6681aa79-a2f9-4ec0-bdfc-cb9788646548": {
        "issue_category": "Feature Problem",
        "specific_issue": "Filters not working",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "4c63f946-68ff-42da-9aef-d694d4c3c6d3": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

     "3ec3dcf2-6990-4229-a2cd-ca608f87b625": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "App stops working and cannot send or receive messages or calls",
        "sentiment": "Negative",
        "severity": "High"
    },

    "24e334a6-61a8-4141-a896-008005a64342": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Download problem",
        "sentiment": "Negative",
        "severity": "Medium"
    },

        "8469afc8-f072-4082-a1c5-eb9d570ab045": {
        "issue_category": "Performance",
        "specific_issue": "App is slow, heavy and inefficient",
        "sentiment": "Negative",
        "severity": "Medium"
    },

        "6082b3c5-3f83-47c4-a85d-09855d889be8": {
        "issue_category": "Feature Problem",
        "specific_issue": "Two years of Memories are not showing",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "c5f1c72b-0545-45ce-9976-8e0c45035f47": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "5d008cfd-a1d4-4d53-9f60-de84e7279e51": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on latest updates",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "e27d9b9f-afb6-44df-bb47-ad8b89e044c7": {
        "issue_category": "Account & Login",
        "specific_issue": "Account blocked after long period of inactivity",
        "sentiment": "Negative",
        "severity": "High"
    },

    "039cc623-9a41-4a35-aad5-31783563b34b": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "c5abb0a6-14fa-4c7a-8c46-924c4238dd47": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "45b07735-69a4-413d-a698-0820dff105db": {
        "issue_category": "Account & Login",
        "specific_issue": "New account verification blocked by unresponsive confirmation button",
        "sentiment": "Negative",
        "severity": "High"
    },

    "e9350f0d-8b93-4a28-b272-5114a29ff6c6": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Scam and fraud risk when doing business on Telegram",
        "sentiment": "Negative",
        "severity": "High"
    },

    "3ec2da84-2b69-41a6-8073-7de958aa8875": {
        "issue_category": "Feature Problem",
        "specific_issue": "Poor photo and video quality",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "54f263ed-a2f0-489d-9feb-45efc719668f": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to log in for a week",
        "sentiment": "Negative",
        "severity": "High"
    },

        "e9049aa4-f901-45c7-bba1-9bb19020bcdc": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "a2592b30-448a-4325-816e-5bb7f51aeb64": {
        "issue_category": "Account & Login",
        "specific_issue": "Phone verification call not received",
        "sentiment": "Negative",
        "severity": "High"
    },

     "6d124529-f68f-4456-a8c9-d862fd09db5a": {
        "issue_category": "Feature Problem",
        "specific_issue": "Too many advertisements",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "d0a762be-d29a-4864-bc31-d41d84459056": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to restore the previous lighter dark theme",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "eb33b297-fa97-4807-a920-de452904a6c3": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Repeated crashes and multiple core functions not working",
        "sentiment": "Negative",
        "severity": "High"
    },

    "d50176cd-e1d3-4d3a-85a8-6f4c3881f7ce": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Call audio and sound management problems",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "2d785995-f854-4267-805b-f734cd992971": {
        "issue_category": "Other / General",
        "specific_issue": "General unrelated comment",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "947fcd83-7a28-403d-b03b-d0bc20f52e0e": {
        "issue_category": "Other / General",
        "specific_issue": "Positive comparison with WhatsApp",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "966ae3d9-7510-415a-a8e4-7f0449060de3": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Positive feedback on messaging security",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "1c704d56-3bdb-4c2e-a36e-20099165f829": {
        "issue_category": "Feature Problem",
        "specific_issue": "Snaps cannot be backed up",
        "sentiment": "Negative",
        "severity": "High"
    },

        "2cbc78f8-0609-4f2f-9d09-d0a0a0174a68": {
        "issue_category": "Feature Problem",
        "specific_issue": "Call and message notifications not working and messages not syncing across devices",
        "sentiment": "Negative",
        "severity": "High"
    },

    "5e57508b-1d1a-46be-a50f-9b8cbee7ec19": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to sign up without verification from an existing WeChat user",
        "sentiment": "Negative",
        "severity": "High"
    },

     "65c909dc-9a6f-447b-b3a2-431caefdaecc": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "App cannot be opened",
        "sentiment": "Negative",
        "severity": "High"
    },

    "d95d4c88-2135-418e-a45f-1bce5cf29e84": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to remove Channels feature",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "35a7609f-e42a-4932-a4fd-c1345c3e10b6": {
        "issue_category": "Performance",
        "specific_issue": "Excessive storage usage",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "d69c423a-4c56-4b30-8a53-2342dd0413f9": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "bfaa4500-43e4-47dd-a5a2-ec822a8e1728": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "428b7444-7ba8-403b-9a34-066300580a34": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "e53100e4-e92e-4194-a8fe-072fd24fe983": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on app smoothness",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "bdc9fe37-414e-49b5-9236-38aed85497f7": {
        "issue_category": "Account & Login",
        "specific_issue": "Phone number banned for unknown reason",
        "sentiment": "Negative",
        "severity": "High"
    },

    "90bd4ccf-c032-4418-99be-cbff3c522004": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Recurring glitches despite updates",
        "sentiment": "Negative",
        "severity": "Medium"
    },

        "2fe45dd0-8df0-4797-a26b-5e055e1a6dc2": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on remittance feature and exchange rates",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "a86249b1-af51-47c0-9375-3b89d65eb93e": {
        "issue_category": "Feature Problem",
        "specific_issue": "Excessive ads and poor video and audio quality",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "810096e4-18ec-4fd6-86c0-b826f7137d90": {
        "issue_category": "Feature Problem",
        "specific_issue": "Message and call notifications not working",
        "sentiment": "Negative",
        "severity": "High"
    },

    "8dcb548d-3133-43b1-a665-446518fbdbbb": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "29baad40-32e9-4877-aaa3-2058ec111319": {
        "issue_category": "Account & Login",
        "specific_issue": "Verification code not working and preventing sign-in",
        "sentiment": "Negative",
        "severity": "High"
    },

    "0fcae42d-11d4-4fe3-a3b1-a0ae95c216bd": {
        "issue_category": "Feature Request",
        "specific_issue": "Request to retain the Lite version instead of forcing the full app",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "108aa93e-f6fd-4312-80d3-a26afee7e038": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "App hangs after update",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "e91ebb32-1fc9-4aba-a6eb-28a824de27c4": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Messages fail to load beyond the first few messages",
        "sentiment": "Negative",
        "severity": "High"
    },

    "f1464d69-66cd-4c66-a409-c1cf6d595552": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "75933780-a987-435e-9a0d-3ecce2474749": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Positive feedback on app safety",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "245e7da6-3bb8-46c2-ba03-2570c80b65d0": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to sign in because SMS cannot be sent",
        "sentiment": "Negative",
        "severity": "High"
    },

     "51fc632e-d684-41ff-9bdd-8f9eda0d9e6e": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "81fdbb69-18b1-43b0-9caf-fba99e2ad4bd": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on everyday communication",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "94eedd08-cb4d-4e42-8aec-caa529a20c56": {
        "issue_category": "Other / General",
        "specific_issue": "General uncertain feedback",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "09e1916f-b947-41b5-892b-b3fb260819e4": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "37a3e29a-d749-494a-9e23-4c53cbf7f52d": {
        "issue_category": "Feature Problem",
        "specific_issue": "Too many advertisements",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "2aafba4c-6ebf-4618-b398-b2c04c4295d2": {
        "issue_category": "Privacy / Security",
        "specific_issue": "View-once media saved to public gallery",
        "sentiment": "Negative",
        "severity": "High"
    },

    "8585ca9e-b76b-481e-a727-87cca91a0fc2": {
        "issue_category": "Feature Problem",
        "specific_issue": "Camera roll access not recognized despite permission",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "a0db6535-5eaa-456c-95eb-f8f6dfdb5a85": {
        "issue_category": "Feature Problem",
        "specific_issue": "Unable to send or receive streaks",
        "sentiment": "Negative",
        "severity": "Medium"
    },

        "d075e3e8-89f7-4f24-ac5e-84d07f672590": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "0084e1c5-b541-470c-9d81-9736d319badd": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "54b9468f-9cfe-425e-8aca-ae1e97a78669": {
        "issue_category": "Account & Login",
        "specific_issue": "Unable to log in or sign up",
        "sentiment": "Negative",
        "severity": "High"
    },

    "493e2d94-7ce8-44c1-a42e-344997c5ba81": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on communication",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "a445103a-5a81-46cf-b2c0-1b0f0c9ecbd1": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

    "617b66c1-95ab-49a2-b600-fadf2313d9a2": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "a9aa4432-61b6-448d-a4a4-91ddf159ddad": {
        "issue_category": "Other / General",
        "specific_issue": "General usage statement",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "f9d52913-c921-4527-831b-bd27e304ef21": {
        "issue_category": "Account & Login",
        "specific_issue": "New account phone number blocked as spam",
        "sentiment": "Negative",
        "severity": "High"
    },

    "499f72d1-a0b5-41ad-bc0c-82c86f9e8d40": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on Wi-Fi reliability",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "14d3b9a7-8dc6-4b79-a9c7-ab4b3da54d4f": {
        "issue_category": "Feature Problem",
        "specific_issue": "Filters repeatedly not working",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "c29abb43-e7b1-4aa3-b3e0-687519b4e60e": {
        "issue_category": "Other / General",
        "specific_issue": "General dissatisfaction",
        "sentiment": "Negative",
        "severity": "Low"
    },

        "fc97a776-22cb-4fa4-93fe-b5b3aa961b9c": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on communication effectiveness",
        "sentiment": "Positive",
        "severity": "Low"
    },
        "9acc8a10-170a-4399-a586-0b44b0967377": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on customer support",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "0c86e777-22e9-4124-8b21-3d3b6f8aceac": {
        "issue_category": "Performance",
        "specific_issue": "Excessive storage usage",
        "sentiment": "Negative",
        "severity": "Medium"
    },

    "cf0e3eb5-4d72-44c1-a7f1-f8f8e1613654": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "9423a7b8-c0a7-4bb4-8af5-33cba968040f": {
        "issue_category": "Privacy / Security",
        "specific_issue": "Scam and fraud risk on the platform",
        "sentiment": "Negative",
        "severity": "High"
    },

    "265013a2-8444-4825-b188-291a015b6f93": {
        "issue_category": "Other / General",
        "specific_issue": "General usage intent",
        "sentiment": "Neutral",
        "severity": "Low"
    },

    "ae5d0be9-70a9-4c57-a0f5-0d0502de24fd": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "5925811c-dcda-4d16-a689-ca56b3c4ae70": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

    "d7f6ae0b-01d4-4518-8f19-49f6d7514a92": {
        "issue_category": "Other / General",
        "specific_issue": "Positive feedback on convenience",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "3f4d72ec-123d-4ae3-a82d-1a61dcf734e3": {
        "issue_category": "Feature Request",
        "specific_issue": "Request for ability to mute stories",
        "sentiment": "Neutral",
        "severity": "Low"
    },

        "b36ca69c-0bfe-4665-a658-7c987ffb439a": {
        "issue_category": "Other / General",
        "specific_issue": "General positive feedback",
        "sentiment": "Positive",
        "severity": "Low"
    },

        "611cfaf8-2635-4086-a012-5a61f34ab36c": {
        "issue_category": "Bugs & Technical Issues",
        "specific_issue": "Messenger repeatedly crashes on device",
        "sentiment": "Negative",
        "severity": "Medium"
    },
}


updated = 0

for review_id, labels in annotations.items():

    mask = df["review_id"] == review_id

    if mask.any():

        df.loc[mask, "issue_category"] = labels["issue_category"]
        df.loc[mask, "specific_issue"] = labels["specific_issue"]
        df.loc[mask, "sentiment"] = labels["sentiment"]
        df.loc[mask, "severity"] = labels["severity"]

        updated += 1

    else:
        print("Warning: review_id not found:", review_id)


df.to_csv(
    FILE_PATH,
    index=False,
    encoding="utf-8"
)


print("Annotation batch completed!")
print("Rows updated:", updated)