#include "qwen3.hpp"

std::vector<struct infer_ctx> inputs;
void load_inputs()
{
    int total_records;
    FILE* fp = fopen("inputs.bin", "rb");

    fread(&total_records, 4, 1, fp);
    for(int i=0;i<total_records;i++)
    {
        struct infer_ctx input;
        memset(&input, 0, sizeof(input));
        fread(&input.seqlen, 4, 1, fp);
        fread(input.input_ids, 4, input.seqlen, fp);
        inputs.push_back(input);
    }
    fclose(fp);
}

int main()
{
	initamx_tiles();
	init_model_weight();
	load_inputs();

    auto t1 = std::chrono::high_resolution_clock::now();
	qwen_model_infer(inputs[1]);
    auto t2 = std::chrono::high_resolution_clock::now() - t1;
    printf("infr timespan: %fms\n", t2.count()/1000000.0f);

	return 0;
}