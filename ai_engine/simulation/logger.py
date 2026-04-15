def log_conversation_analysis(conversation, scenario):
    print("\n=== CONVERSATION TRACE ANALYSIS ===\n")

    for turn in conversation.turns:
        print(f"[Turn {turn.turn_id}] {turn.role.upper()}: {turn.message}")

        # Match with expected state
        expected_state = conversation.memory.current_state

        print(f"  → Expected State: {expected_state}")

        # Check slots
        if turn.slots_extracted:
            print(f"  → Slots Extracted: {turn.slots_extracted}")

        # Check intent
        if turn.intent_detected:
            print(f"  → Intent Detected: {turn.intent_detected}")

        print("")

    print("=== FINAL MEMORY ===")
    print(conversation.memory)  