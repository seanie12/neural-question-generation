import collections
import json
import re
import string
import sys

import torch
from pytorch_pretrained_bert import BertForQuestionAnswering, BertTokenizer
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset

import config
from squad_utils import read_squad_examples, convert_examples_to_features, write_predictions


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
test_file = "./squad/new_test-v1.1.json"
eval_examples = read_squad_examples(test_file, is_training=False, debug=False)
eval_features = convert_examples_to_features(eval_examples,
                                             tokenizer=tokenizer,
                                             max_seq_length=config.max_seq_len,
                                             max_query_length=config.max_query_len,
                                             doc_stride=128,
                                             is_training=False)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0))
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)

model = BertForQuestionAnswering.from_pretrained("./save/dual/train_507200353/bert_1_2.958")
model = model.to(config.device)
device = "cuda:2"
model.eval()
all_results = []
for data in eval_dataloader:
    input_ids, input_mask, segment_ids, example_indices = data
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
    for i, example_index in enumerate(example_indices):
        start_logits = batch_start_logits[i].detach().cpu().tolist()
        end_logits = batch_end_logits[i].detach().cpu().tolist()
        eval_feature = eval_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        all_results.append(RawResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits))

output_prediction_file = "./result/qa/dual_predictions.json"
output_nbest_file = "./result/qa/dual_nbest_predictions.json"
output_null_log_odds_file = "./result/qa/dual_null_odds.json"
write_predictions(eval_examples, eval_features, all_results,
                  n_best_size=20, max_answer_length=30, do_lower_case=True,
                  output_prediction_file=output_prediction_file,
                  output_nbest_file=output_nbest_file,
                  output_null_log_odds_file=output_null_log_odds_file,
                  verbose_logging=False,
                  version_2_with_negative=False,
                  null_score_diff_threshold=0)

with open("./squad/new_test-v1.1.json") as dataset_file:
    dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
with open("./result/qa/dual_predictions.json") as prediction_file:
    predictions = json.load(prediction_file)
print(json.dumps(evaluate(dataset, predictions)))
