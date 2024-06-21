from model import *

input_size = 2
hidden_size = 4
output_size = 2

model_inference = LSTMModel(input_size, hidden_size, output_size)
model_inference.load_state_dict(torch.load('saved_model_lstm.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inference.to(device)

test_input = torch.tensor([[150, 19]], dtype=torch.float32).to(device)


model_inference.eval()
with torch.no_grad():
    predicted_probs = model_inference(test_input)
    _, predicted_labels = torch.max(predicted_probs, 1)

print("Predicted class:", predicted_labels.item())