from rag_engine import rag

def main():
    # Initialize the RAG engine
    qa = rag()

    # Example query
    query = input("Enter your question: ")
    
    # Get the answer
    answer = qa(query)
    
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
 