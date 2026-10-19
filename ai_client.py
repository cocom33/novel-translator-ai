
import os
import argparse
from pathlib import Path
from google import genai
from google.genai import types


def read_prompt(value):
    """Baca prompt dari file jika --prompt-file digunakan."""
    return Path(value).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Gemini AI Client"
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Nama model Gemini yang ingin digunakan"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt yang dikirim ke AI"
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path file berisi prompt"
    )
    parser.add_argument(
        "--system",
        default="Kamu adalah asisten AI yang membantu.",
        help="System instruction"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Aktifkan mode percakapan"
    )

    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "API key belum diatur. Set GEMINI_API_KEY terlebih dahulu."
        )

    if args.prompt and args.prompt_file:
        raise SystemExit(
            "Gunakan --prompt atau --prompt-file, bukan keduanya."
        )

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=args.system,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )

    try:
        if args.chat:
            chat = client.chats.create(
                model=args.model,
                config=config,
            )
            print(f"Model aktif: {args.model}")
            print("Ketik /exit untuk keluar.")

            while True:
                prompt = input("\nKamu: ").strip()

                if prompt.lower() == "/exit":
                    break
                if not prompt:
                    continue

                response = chat.send_message(prompt)
                print("\nAI:", response.text or "[Tidak ada teks]")
        else:
            if args.prompt_file:
                prompt = read_prompt(args.prompt_file)
            else:
                prompt = args.prompt

            if not prompt:
                prompt = input("Masukkan prompt: ")

            response = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config=config,
            )
            print(response.text or "[Tidak ada teks]")

    except Exception as exc:
        print(f"Request gagal: {exc}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
