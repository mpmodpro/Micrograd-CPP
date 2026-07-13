#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>
#include <random>


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
		
		
	private:
		Value(float data, const std::string &op, size_t id) : data(data) , op(op), id(id) {};	
	public:
		static ValuePtr create(float data, const std::string &op = ""){
			return std::shared_ptr<Value>(new Value(data, op, Value::currentID++));
		}
		
		~Value(){
			Value::currentID--;
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
			(*it)->print();
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
			for(size_t in = 0; in <= nin; in++)
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
	
	ValuePtr operator()(std::vector<ValuePtr>& x){
		if(weights.size() != x.size())
			throw std::invalid_argument("Inputs & Neuron Weights need to be of the same size");
			
		ValuePtr sum = Value::create(0.0);
		for(size_t idx = 0; idx <= weights.size(); ++idx)
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
			for(size_t idx = 0; idx <= noN; ++idx)
				this->neurons.emplace_back(Neuron(dimN, act));
		} 
		
		ValuePtr operator()(const std::vector<ValuePtr>& x){
			std::vector<ValuePtr> out;
			
		}
};

int main(){
	auto a = Value::create(1.0, "");
	auto b = Value::create(2.0, "");
	
	auto c = Value::add(a, b);
	
	auto d = Value::multiply(c, c);
	
	auto loss = Value::add(d,d);
	
	loss->backprop();
}