#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>
#include <random>
#include <chrono>


class Value; 

using ValuePtr = std::shared_ptr<Value>;

struct Hash{
	size_t operator()(const ValuePtr value) const;
};

class Value: public std::enable_shared_from_this<Value>{

	public:
		inline static size_t currentID;
		float data;
		float grad;
		std::string op;
		size_t id;
		std::vector<ValuePtr> prev;
		std::function<void()> backward;
		float velocity = 0.0f;
		
		
	private:
		Value(float data, const std::string &op, size_t id) : data(data) , op(op), id(id) {};	
	public:
		static ValuePtr create(float data, const std::string &op = ""){
			return std::shared_ptr<Value>(new Value(data, op, Value::currentID++));
		}
		
		~Value(){
			//Value::currentID--;
		}
		
	static ValuePtr add(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data + rhs->data, "+");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad;
			rhs_weak.lock()->grad += out_weak.lock()->grad;
		};
		return out;
	}
	
	static ValuePtr substract(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data - rhs->data, "-");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad;
			rhs_weak.lock()->grad -= out_weak.lock()->grad; 
		};
		return out;
	}
	
	static ValuePtr multiply(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data * rhs->data, "*");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad * rhs_weak.lock()->data;
			rhs_weak.lock()->grad += out_weak.lock()->grad * lhs_weak.lock()->data;
		};
		return out;
	}
	
	static ValuePtr pow(const ValuePtr &base, float exponent){
		auto out = Value::create(std::pow(base->data, exponent), "^");
		out->prev = {base};
		out->backward = [base_weak = std::weak_ptr<Value>(base), out_weak = std::weak_ptr<Value>(out), exponent](){
			base_weak.lock()->grad += out_weak.lock()->grad * exponent * std::pow(base_weak.lock()->data, exponent - 1);
		};
		return out;
	}
	
	static ValuePtr divide(const ValuePtr &lhs, const ValuePtr &rhs){
		auto reciprocal = pow(rhs, -1);
		return multiply(lhs, reciprocal);
		
	};
	
	
	static ValuePtr relu(const ValuePtr &input){
		float val = std::max(0.0f, input->data);
		auto out = Value::create(val, "ReLU");
		out->prev = {input};
		out->backward = [input_weak = std::weak_ptr(input), out_weak = std::weak_ptr(out)](){
			//out grad * inward gradient
			input_weak.lock()->grad += (out_weak.lock()->data > 0) * out_weak.lock()->grad;
		};
		return out;
	}
	
	static ValuePtr sigmoid(const ValuePtr & input){
		float d = input->data;
		float sig = 1 / (1 + std::exp(-d));
		auto out = Value::create(sig, "Sigmoid");
		out->prev = {input};
		out->backward = [input_weak = std::weak_ptr(input), out_weak = std::weak_ptr(out), sig](){
			//out grad * inward gradient
			input_weak.lock()->grad += out_weak.lock()->grad * (sig * (1 - sig));
		};	
		return out;
	}
	
	
	void buildtopo(ValuePtr v, std::unordered_set<ValuePtr, Hash>& visited, std::vector<ValuePtr>& topo){
		if(visited.find(v) == visited.end()){
			visited.insert(v);
			for(const auto& child : v->prev)
				buildtopo(child, visited, topo);
			topo.push_back(v);
		}
	}
	
	void backprop(){
		std::unordered_set<ValuePtr, Hash> visited;
		std::vector<ValuePtr> topo;
		buildtopo(shared_from_this(), visited, topo);
		
		this->grad = 1.0f;
		
		for(auto it = topo.rbegin(); it != topo.rend(); it++){
			if((*it)->backward)
				(*it)->backward();
			//(*it)->print();
		}
	}
	
	void print(){
		std::cout << "[Data: " << this->data << "\t Grad: " << this->grad << "]" << std::endl;
	}
};

size_t Hash::operator()(const ValuePtr value) const{
	return std::hash<std::string>()(value->op) ^ std::hash<float>()(value->data);
}


enum ActivationType{
	RELU,
	SIGMOID
};

class Activation{
	public:
		static ValuePtr Relu(const ValuePtr &input){
			return Value::relu(input);
		}
		
		static ValuePtr Sigmoid(const ValuePtr &input){
			return Value::sigmoid(input);
		}
		
		static inline std::unordered_map<ActivationType, std::function<ValuePtr(ValuePtr&)>> actfunc{
			{ActivationType::RELU, Relu},
			{ActivationType::SIGMOID, Sigmoid}
		};
};

float getRandomFloat(){
	static std::random_device rd;
	static std::mt19937 rng(rd());
	static std::uniform_real_distribution dis(-1.0f,1.0f);
	return dis(rng);
}




class Neuron{
	private:
		std::vector<ValuePtr> weights;
		ValuePtr bias = Value::create(0.0, "");
		const ActivationType act;
	
	public:
		Neuron(size_t nin, const ActivationType act) : act(act){
			for(size_t in = 0; in < nin; in++)
				weights.emplace_back(Value::create(getRandomFloat()));
		}
		
	
	
	void zeroGrad(){
		for(auto weight : weights)
			weight->grad = 0;
		bias->grad = 0;
	}
	
	 std::vector<ValuePtr> Parameters() const{
		std::vector<ValuePtr> out;
		out.reserve(weights.size() + 1);
		out.insert(out.end(), weights.begin(), weights.end());
		out.push_back(bias);
		
		return out;
	}
	
	void printParameters() const{
		printf("No. of Parameters: %zu\n", weights.size()+1);
		for(const auto& params: weights)
			printf("%f, %f\n", params->data, params->grad);
		printf("%f, %f\n", bias->data, bias->grad);
		printf("\n");
	}
	
	ValuePtr operator()(const std::vector<ValuePtr>& x) const{
		if(weights.size() != x.size())
			throw std::invalid_argument("Inputs & Neuron Weights need to be of the same size");
			
		ValuePtr sum = Value::create(0.0);
		for(size_t idx = 0; idx < weights.size(); ++idx)
			sum = Value::add(sum, Value::multiply(weights[idx], x[idx]));
		
		sum = Value::add(sum, bias);
		
		const auto& activationFun = Activation::actfunc.at(act);
		
		return activationFun(sum);
	}
};

class Layer{
	public:
		std::vector<Neuron> neurons;
		Layer(size_t dimN, size_t noN, const ActivationType& act){
			for(size_t idx = 0; idx < noN; ++idx)
				this->neurons.emplace_back(Neuron(dimN, act));
		} 
		
		std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x){
			std::vector<ValuePtr> out;
			out.reserve(neurons.size());
			std::for_each(neurons.begin(), neurons.end(), [&out, &x](const auto& neuron){
				out.emplace_back(neuron(x));
			});

			return out;
		}
		
		void zeroGrad(){
			for(auto& neuron : neurons)
				neuron.zeroGrad();
		}
		
		std::vector<ValuePtr> Parameter() const{
			std::vector<ValuePtr> params;
			if (params.empty())
				for(const auto& neuron : neurons)
					for(const auto& p : neuron.Parameters())
						params.push_back(p);
					
			return params;
		}
		
		void printParameters(){
			const auto& params = this->Parameter();
			printf("No. of Parameters: %zu\n", params.size());
			for(const auto& p: params){
				std::cout << &p << " ";
				printf("%f, %f\n", p->data, p->grad);
			}
			printf("\n");	
		}
		
};

class Tensor{
private:
	std::vector<std::vector<ValuePtr>> tensor;

public:
	Tensor(std::initializer_list<float> input){
		for(const auto& elem : input){
			std::vector<ValuePtr> subtensor;
			subtensor.push_back(Value::create(elem));
			tensor.push_back(subtensor);
		}
	}
	
	auto begin(){
		return tensor.begin();
	}
	
	auto end(){
		return tensor.end();
	}
	
	auto begin() const{
		return tensor.begin();
	}
	
	auto end() const{
		return tensor.end();
	}
	
	//i th row
	std::vector<ValuePtr> operator[](const size_t idx) const{
		if(idx >= tensor.size())
			throw std::invalid_argument("Index out of bound");
		
		return tensor[idx];
	}
	
	//j th row
	ValuePtr at(const size_t idx, const size_t jdx) const{
		if(idx >= tensor.size() || jdx >= tensor[idx].size())
			throw std::invalid_argument("Index out of bound");
		
		return tensor[idx][jdx];
	}
	
	auto reset(){
		tensor.clear();
	}
	
	auto push_back(const std::vector<ValuePtr> elems){
		std::vector<ValuePtr> subtensor;
		std::copy(elems.begin(), elems.end(), std::back_inserter(subtensor));
		this->tensor.emplace_back(subtensor);
	}
	
	size_t size() const{
		return tensor.size();
	}
	
	void zeroGrad(){
		for(auto& subtensor : tensor)
			for(auto& value: subtensor)
				value->grad = 0;
	}
};


class MLP{
private:
	std::vector<size_t> sizes;
	std::vector<Layer> layers;
	float learning_rate;
	
public:	
	MLP(size_t nin, std::vector<size_t> nouts, const float learning_rate) : learning_rate(learning_rate){
		sizes.reserve(4);
		sizes.push_back(nin);
		std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes));
		
		for(size_t idx = 0; idx < sizes.size() - 1; ++idx){
			layers.emplace_back(sizes[idx], sizes[idx+1], ActivationType::SIGMOID);
		}
	}
	
	std::vector<ValuePtr> Parameter() const{
		std::vector<ValuePtr> params;
		if (params.empty())
			for(const auto& layer : this->layers)
				for(const auto& p : layer.Parameter())
					params.push_back(p);
					
		return params;
	}
	
	void printParameters(){
			const auto& params = this->Parameter();
			printf("No. of Parameters: %zu\n", params.size());
			for(const auto& p: params){
				std::cout << &p << " ";
				printf("%f, %f\n", p->data, p->grad);
			}
			printf("\n");	
		}
		
	void zeroGrad(){
			for(auto& layer : this->layers)
				layer.zeroGrad();
		}
	
	void gradientDescent(){
		for(auto& p : this->Parameter())
			p->data += ((-learning_rate) * p->grad);
	}
	
	void gradientDescentwithMomentum(float Beta){
		for(auto& p : this->Parameter()){
			p->velocity = (Beta * p->velocity) + (learning_rate * p->grad);
			p->data = p->data - p->velocity;
 		}
	}
	
	void beginNAG(float Beta){
		for(auto& p : this->Parameter()){
			p->data -= (Beta * p->velocity);
		}
	}
	
	void endNAG(float Beta){
		for(auto& p : this->Parameter()){
			p->data += (Beta * p->velocity);
		}
	}
	
	void NAG(float Beta){
		for(auto& p : this->Parameter()){
			p->velocity = (Beta * p->velocity) + (learning_rate * p->grad);
			p->data = p->data - p->velocity;
		}
	}
	
	std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& input){
		std::vector<ValuePtr> x = input;
		
		for(auto& layer : layers){
			std::vector<ValuePtr> y = layer(x);
			x = y;
		}
		
		return x;
	}
	
};


int main(){
	Tensor input({0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.333333, 0.333333, 0.529412, 0.666275, 0});

	Tensor output({0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.333333, 0.333333, 0.529412, 0.666275, 0});
	
	
	const float LearningRate = 0.01;
	
	size_t nwSize = input[0].size();
	
	std::cout << "Input n/w size: " << nwSize << std::endl;
	
	auto mlp = std::unique_ptr<MLP>(new MLP(nwSize ,{16, 16, nwSize} , LearningRate));
	
	auto start_time_point = std::chrono::high_resolution_clock::now();
	
	Tensor ypred({});
	
	for(size_t idx = 0; idx < 1000000; ++idx){
		std::cout << "//////////////////////////////////////////////////////////////////////////" << std::endl;
		
		input.zeroGrad();
		
		auto loss = Value::create(0.0f);
		
		mlp->zeroGrad();
		
		mlp->beginNAG(0.9);
		
		for(const auto& inp : input)
			ypred.push_back((*mlp)(inp));
			
		//Loss
		
		for(size_t idx = 0; idx < output[0].size(); ++idx){
			auto c = Value::substract(output.at(0, idx), ypred.at(0, idx));
			auto d = Value::multiply(c, c);
			
			loss = Value::add(loss, d);
		}		
		
		loss->backprop();
		
		mlp->endNAG(0.9);
		
		mlp->NAG(0.9);
		
		std::cout << "Iteration: " << idx << " " << "Loss: " << loss->data << " , " << Value::currentID << std::endl;
		
		ypred.reset();
	}
}