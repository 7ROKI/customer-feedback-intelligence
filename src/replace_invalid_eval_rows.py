import pandas as pd


eval_df = pd.read_csv("data/formal_evaluation_set.csv")
clean_df = pd.read_csv("data/cleaned_reviews.csv")


# Permanently blacklisted review IDs
blacklisted_ids = {
    "f02fb629-0175-4340-8b00-7bfc59dfb214",
    "efc5a720-a120-43d1-9552-f1a64c72430f",
    "028c1a81-2079-4fc9-8786-89addb54cae4",
    "150be953-1484-4861-bdc4-f941586b4d81",
    "f13aa56a-28dc-4e3f-8e97-3c1b8ef49def",
    "407c6a57-fd61-4d00-979e-ed78a0713d0b",
    "0f3748ee-e78e-452c-92d3-b68c683b8cbc",
    "a34d544d-e8c2-4c00-bdff-035d21c74ac7",
    "2e4ad38e-5312-49c7-adaf-23177f3d296d",
    "c3048fc2-14de-4b17-998a-19b710f03355",
    "e3f28a51-3024-4509-b9d1-3de4097d9fe8",
    "1e724e33-09bc-4758-b543-99e8ec77f68e",
    "55e0c45a-5047-4ed3-acfa-3c44563ad5a7",
    "6cc38978-1e9f-4db0-ae37-4b1db02d3347",
    "64169dd8-d0ab-4db6-b369-5170177c0335",
    "40699181-9af4-47a1-8cb6-416f7041b8be",
    "41afd0ce-a743-4cac-ab4a-006bb82422bf",
    "80bb7e7f-0979-4a79-9c1c-9ea8ea0fc58a",
    "3d8db477-07c9-49d5-af04-b78995c9f524",
    "ab15282f-86b7-4052-b6a8-ba8a2d64bce2",
    "a02799cd-f450-45b4-9631-a771ee756af8",
    "9c4bff21-97da-49f8-9738-3ef3c3dbfe2e",
    "01eda65b-b113-4100-93e0-d33f6892b4ea",
    "a782e1cc-af91-4725-b923-7dbed91e68d3",
    "4e042cff-877d-4e12-9b97-795f0d1148d0",
    "07b2d03d-8bfc-4dab-bf44-2233396bc8f4",
    "6c472c0e-1aa4-4424-8f50-8ae336a9046c",
    "76849602-502f-4b4d-b89f-fc2d532eb76d",
    "0e6eb88f-572a-40ca-972b-3e222113f819",
    "97d5fe55-6bc0-4b5a-b45a-321d8939eb4b",
    "97962566-d71a-4d55-bbce-edfe696a5dda",
    "7d6fcfd2-fdca-47c7-80f0-e0b1d90d515c",
    "31f66254-839f-42d2-b3cc-8499b7fe0664",
    "cab6b9ae-bb6c-4176-9c3e-37936fe5b20d",
    "ff2303ad-bff1-4bd5-9a97-93b8ddc96e09",
    "d82c5efb-18d6-4f81-8da9-a00c308c9373",
    "e0dc239f-d797-40c0-b029-0f8aacb30d02",
    "d78b045d-3ad4-41f1-92ec-c553be7b2447",
    "99dfb17a-b047-4999-9152-07824b017bc3",
    "b0556d15-4a71-4750-b5bc-0c3b33292d3d",
    "290e5859-e814-4c18-a6a2-e759391e4eac",
    "cc9dcc91-68e8-4599-af6a-2074bb0486ff",
    "815ac53c-2fc6-4184-82dc-c3c374ba7b4b",
    "5b7df92f-1295-40cf-91b2-92e32e88b724",
    "712dd089-b09f-4d2b-9f89-c010eaf975d7",
    "8f90d361-3a25-46cf-997e-4b7a6f794fb0",
    "94ec28be-4259-457a-8e2f-469b1ae3d9f8",
    "11e2f8d6-3e33-458c-bd5a-af01599482eb",
    "c9d080dc-86be-4de0-8e13-3781517969df",
    "508b5e2a-23de-4af3-9bb4-8c515dddda53",
    "4a4f1784-a055-489c-9491-0ba6418be3b0",
    "e63de48f-c2b8-4a80-9196-fb5710790584",
    "7d2b0897-5ca1-4126-b26d-a49dbff0a6e4",
    "014a929e-6009-42ee-b826-b88bac275eb2",
    "18ac030a-a196-4741-ab06-2895d7659689",
    "4ccbf48f-a52d-469a-acc6-6606df7c3164",
    "adf257da-9fbf-4658-b978-762c065162a0",
    "220b203f-6b62-4541-8963-5300d27045da",
    "a38a8f9d-76bd-480a-8391-fc0f1838a4a4",
    "bfb7ad17-f90d-40e8-97ef-2eee4f6c6ae6",
    "de94b8a7-7429-4f49-a568-7ba869f111f5",
    "2e5a393c-9ee8-4940-b1e7-7f6f7492da22",
    "48faa965-45b2-442d-9b09-64ec4e33289a",
    "f6556781-7d4a-4421-b823-22db5ebb5b3f",
    "e81a633c-f9fc-4c40-96ff-9fb08ba57418",
    "058171d9-30e0-4389-8d92-3c95447761a6",
    "27dd332c-0960-4266-9c08-5639a3762e0c",
}


# Invalid rows to replace in this run
current_invalid_ids = {
    
    "27dd332c-0960-4266-9c08-5639a3762e0c",
}


invalid_rows = eval_df[
    eval_df["review_id"].isin(current_invalid_ids)
].copy()

print("Invalid rows found:", len(invalid_rows))


if len(invalid_rows) == 0:
    print("No matching invalid rows found. Nothing was changed.")
    raise SystemExit

# Remove them
eval_df = eval_df[
    ~eval_df["review_id"].isin(current_invalid_ids)
].copy()


# IDs that must never be selected
used_ids = set(eval_df["review_id"]) | blacklisted_ids

replacement_rows = []


for _, invalid_row in invalid_rows.iterrows():

    app_id = invalid_row["app_id"]
    rating = invalid_row["rating"]

    # Prefer same app + same rating
    candidates = clean_df[
        (clean_df["app_id"] == app_id) &
        (clean_df["rating"] == rating) &
        (~clean_df["review_id"].isin(used_ids))
    ].copy()

    # Remove obviously very short candidates
    candidates = candidates[
        candidates["feedback"].astype(str).str.len() >= 15
    ]

    # If necessary, relax rating but keep same app
    if len(candidates) == 0:
        candidates = clean_df[
            (clean_df["app_id"] == app_id) &
            (~clean_df["review_id"].isin(used_ids))
        ].copy()

        candidates = candidates[
            candidates["feedback"].astype(str).str.len() >= 15
        ]

    replacement = candidates.sample(
        n=1,
        random_state=100 + len(replacement_rows)
    ).iloc[0]

    replacement_rows.append(replacement)
    used_ids.add(replacement["review_id"])


replacement_df = pd.DataFrame(replacement_rows)

replacement_df["issue_category"] = ""
replacement_df["specific_issue"] = ""
replacement_df["sentiment"] = ""
replacement_df["severity"] = ""

replacement_df = replacement_df[
    [
        "review_id",
        "feedback",
        "rating",
        "date",
        "app_id",
        "issue_category",
        "specific_issue",
        "sentiment",
        "severity"
    ]
]


eval_df = pd.concat(
    [eval_df, replacement_df],
    ignore_index=True
)


eval_df = eval_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


eval_df.to_csv(
    "data/formal_evaluation_set.csv",
    index=False,
    encoding="utf-8"
)


print()
print("Replacement completed!")
print("Final rows:", len(eval_df))

print()
print("New replacement reviews:")

for _, row in replacement_df.iterrows():
    print()
    print("Review ID:", row["review_id"])
    print("App:", row["app_id"])
    print("Rating:", row["rating"])
    print("Feedback:", row["feedback"])
    print("-" * 60)