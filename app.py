    def llm(prompt):
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                               response_mime_type="application/json",
                                               temperature=0.2))
        return resp.text or ""
