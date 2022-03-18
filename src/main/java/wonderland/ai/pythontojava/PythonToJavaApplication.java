package wonderland.ai.pythontojava;

import org.pmml4s.model.Model;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Stream;

@SpringBootApplication
public class PythonToJavaApplication {

	public static void main(String[] args) {
		SpringApplication.run(PythonToJavaApplication.class, args);
	}

	@EventListener(ApplicationReadyEvent.class)
	public void start() {
		predictDiabetes();
	}

	private void predictHousingPrice() {
		final Model model = Model.fromFile("python/housing_price/model.pmml");
		//todo
	}

	private void predictDiabetes() {
		final Model model = Model.fromFile("python/diabetes/model.pmml");
		Map<String, Double> values = Map.of(
				"age", 0.01809694d,
				"sex", 0.00301924d,
				"bmi", 0.00511107d,
				"bp", -0.00222774d,
				"s1", -0.02633611d,
				"s2", -0.02699205d,
				"s3", 0.01550536d,
				"s4", -0.02104282d,
				"s5", -0.02421066d,
				"s6", -0.05492509d
		);
		Object[] valuesMap = Arrays.stream(model.inputNames()).map(values::get).toArray();
		Object[] result = model.predict(valuesMap);
		Stream.of(result).map(Double.class::cast).forEach(System.out::println);
	}
}
