from rag_engine import RAG_Engine

def main():
    

    query = input("Enter your question: ")
    answer = RAG_Engine(query)

    # Get the answer

    
    # Print the answer
    print("Answer:", answer)
if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
 